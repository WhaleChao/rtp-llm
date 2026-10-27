#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/normal_engine/NormalModelInputGatherer.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/distribute/CpuTpBroadcaster.h"

#include <atomic>
#include <chrono>
#include <future>
#include <thread>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <spawn.h>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>

extern char** environ;

namespace rtp_llm {
namespace {

// Always release and drain the callback, including if std::async or a tensor
// operation throws. The callback never outlives its host flag.
struct PendingGpuWork {
    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
    std::atomic<bool> released{false};

    ~PendingGpuWork() {
        released.store(true);
        cudaStreamSynchronize(stream.stream());
    }
};

// Test the real gatherer and model boundary, not a Python reimplementation.
class PrefillHostBlockTableTest: public DeviceTestBase {
public:
    static void SetUpTestSuite() {
        py::initialize_interpreter();
        auto main = py::module_::import("__main__");
        py::class_<torch_ext::PyModelInitResources>(main, "TestModelInitResources");
        py::exec("class TestModel:\n    def initialize(self, resources):\n        return True\n", main.attr("__dict__"));
    }

    static void TearDownTestSuite() {
        py::finalize_interpreter();
    }

    std::unique_ptr<PyWrappedModel> model() {
        GptModelDescription description{};
        description.attention_conf.use_mla = true;
        description.attention_conf.kv_cache_dtype = KvCacheDataType::FP8;
        GptModelInitParams params{{}, description};
        params.hw_kernel_config.enable_cuda_graph = false;
        return std::make_unique<PyWrappedModel>(params, py::module_::import("__main__").attr("TestModel")());
    }

    GptModelInputs gather(bool decode, TensorHolder& holder, int blocks_per_kv = 1) {
        ModelConfig config;
        config.max_seq_len = 2048;
        config.vocab_size = 2048;
        config.num_layers = 2;
        auto query = std::make_shared<GenerateInput>();
        query->input_ids = torch::tensor({1, 2, 3}, torch::kInt32);
        query->generate_config = std::make_shared<GenerateConfig>();
        auto stream = std::make_shared<NormalGenerateStream>(
            query, config, RuntimeConfig{}, ResourceContext{}, nullptr);
        BatchKVCacheResource cache;
        cache.resetBatchSize(1);
        cache.initGroups(2, 2, {0, 1}, blocks_per_kv);
        cache.setBatchBlocks(0, 0, {7, 9});
        cache.setBatchBlocks(0, 1, {11, 13});
        stream->setKVCache(cache);
        stream->setIsContextStream(!decode);
        NormalModelInputGathererConfig gather_config;
        gather_config.num_layers = 2;
        gather_config.vocab_size = 2048;
        gather_config.input_vocab_size = 2048;
        gather_config.seq_size_per_block = 64 * blocks_per_kv;
        gather_config.kernel_seq_size_per_block = 64;
        gather_config.kernel_blocks_per_kv_block = blocks_per_kv;
        gather_config.kv_cache_group_nums = 2;
        gather_config.layer_to_kv_cache_group_id = {0, 1};
        gather_config.kv_cache_group_types = {CacheGroupType::FULL, CacheGroupType::FULL};
        auto result = NormalModelInputGatherer(gather_config).gather(StreamGroups({stream}), holder);
        if (!result.ok()) {
            throw std::runtime_error(result.status().ToString());
        }
        return std::move(result).value();
    }

    static void flush(PyWrappedModel& model) {
        if (model.d2d_copies_.num_copies) {
            fusedCopy(model.d2d_copies_);
            model.d2d_copies_.clear();
        }
        c10::cuda::getCurrentCUDAStream().synchronize();
    }
};

TEST_F(PrefillHostBlockTableTest, GatherKeepsPrefillHostButDecodeDevice) {
    TensorHolder holder;
    auto prefill = gather(false, holder);
    EXPECT_TRUE(prefill.kv_cache_kernel_block_id.is_cpu());
    if (prefill.kv_cache_kernel_block_id.is_cpu()) {
        EXPECT_TRUE(prefill.kv_cache_kernel_block_id.is_pinned());
    }
    EXPECT_TRUE(torch::equal(prefill.kv_cache_kernel_block_id.cpu(),
                            torch::tensor({7, 9, 11, 13}, torch::kInt32).reshape({2, 1, 2})));
    auto decode = gather(true, holder);
    EXPECT_TRUE(decode.kv_cache_kernel_block_id.is_cuda());
    EXPECT_TRUE(torch::equal(decode.kv_cache_kernel_block_id.cpu(), prefill.kv_cache_kernel_block_id.cpu()));
}

TEST_F(PrefillHostBlockTableTest, HostAndDeviceViewsMatchAcrossGroupsAndSteps) {
    auto runner = model();
    for (int step = 0; step < 3; ++step) {
        GptModelInputs input;
        input.input_lengths = torch::tensor({3, 4}, torch::kInt32);
        input.sequence_lengths = torch::empty({0}, torch::kInt32);
        // Non-contiguous batch slice of two groups, as produced by microbatching.
        input.kv_cache_kernel_block_id =
            (torch::arange(24, torch::kInt32) + 100 * step).reshape({2, 3, 4}).pin_memory().narrow(1, 1, 2);
        input.kv_cache_block_id = input.kv_cache_kernel_block_id.slice(2, 0, 2);
        torch_ext::PyAttentionInputs attention;
        runner->setupKVCacheForAttentionInputs(attention, input);
        flush(*runner);
        ASSERT_EQ(attention.kv_cache_kernel_block_id_device_by_group.size(), 2);
        for (int g = 0; g < 2; ++g) {
            EXPECT_TRUE(attention.kv_cache_kernel_block_id_device_by_group[g].is_cuda());
            EXPECT_TRUE(torch::equal(attention.kv_cache_kernel_block_id_device_by_group[g].cpu(),
                                    input.kv_cache_kernel_block_id[g]));
        }
        EXPECT_TRUE(torch::equal(attention.kv_cache_kernel_block_id_host, input.kv_cache_kernel_block_id[0]));
        EXPECT_TRUE(torch::equal(attention.kv_cache_block_id_host, input.kv_cache_block_id[0]));
        EXPECT_TRUE(torch::equal(attention.kv_cache_block_id_device.cpu(), input.kv_cache_block_id[0]));
        runner->releaseBuffers();
    }
}

TEST_F(PrefillHostBlockTableTest, Physical512Kernel64LayoutIsUnchanged) {
    TensorHolder holder;
    auto input = gather(false, holder, 8);
    auto runner = model();
    torch_ext::PyAttentionInputs attention;
    runner->setupKVCacheForAttentionInputs(attention, input);
    flush(*runner);
    const auto expected = torch::cat({torch::arange(56, 64, torch::kInt32),
                                      torch::arange(72, 80, torch::kInt32)}).reshape({1, 16});
    EXPECT_TRUE(torch::equal(attention.kv_cache_kernel_block_id_host, expected));
    EXPECT_TRUE(torch::equal(attention.kv_cache_kernel_block_id_device.cpu(), expected));
    EXPECT_TRUE(torch::equal(attention.kv_cache_block_id_host,
                            torch::tensor({7, 9}, torch::kInt32).reshape({1, 2})));
}

TEST_F(PrefillHostBlockTableTest, InputPreparationDoesNotWaitForPendingGpuWork) {
    TensorHolder holder;
    auto input = gather(false, holder);
    auto runner = model();
    // Warm allocations before introducing the stream dependency. This is a
    // bounded fault injection, NOT a claim to reproduce the online NCCL cycle.
    { torch_ext::PyAttentionInputs warm; runner->setupKVCacheForAttentionInputs(warm, input); flush(*runner); }
    PendingGpuWork pending;
    ASSERT_EQ(cudaLaunchHostFunc(pending.stream.stream(), [](void* arg) {
        auto* flag = static_cast<std::atomic<bool>*>(arg);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (!flag->load() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }, &pending.released), cudaSuccess);
    auto prepared = std::async(std::launch::async, [&] {
        c10::cuda::CUDAStreamGuard guard(pending.stream);
        torch_ext::PyAttentionInputs attention;
        runner->setupKVCacheForAttentionInputs(attention, input);
        return attention;
    });
    const bool ready = prepared.wait_for(std::chrono::milliseconds(500)) == std::future_status::ready;
    pending.released.store(true);
    auto attention = prepared.get();
    flush(*runner);
    EXPECT_TRUE(ready) << "Model input preparation waited on GPU work before entering the model";
    EXPECT_TRUE(torch::equal(attention.kv_cache_kernel_block_id_host, input.kv_cache_kernel_block_id[0].cpu()));
    EXPECT_TRUE(attention.kv_cache_kernel_block_id_device.is_cuda());
}

TEST_F(PrefillHostBlockTableTest, EmptyHostTableAndGraphHostGating) {
    auto runner = model();
    for (bool empty : {false, true}) {
        for (bool warmup : {false, true}) {
            for (bool verify : {false, true}) {
                GptModelInputs input;
                input.input_lengths = torch::ones({empty ? 0 : 2}, torch::kInt32);
                input.sequence_lengths = torch::empty({0}, torch::kInt32);
                input.kv_cache_kernel_block_id = torch::zeros({2, empty ? 0 : 2, 4}, torch::kInt64).pin_memory();
                input.warmup = warmup;
                input.is_target_verify = verify;
                runner->enable_cuda_graph_ = true; // Test the real setup gate without capturing a model.
                torch_ext::PyAttentionInputs attention;
                runner->setupKVCacheForAttentionInputs(attention, input);
                flush(*runner);
                EXPECT_TRUE(attention.kv_cache_kernel_block_id_device.is_cuda());
                EXPECT_EQ(attention.kv_cache_kernel_block_id_device.scalar_type(), torch::kInt32);
                // An empty input is classified as decode by the existing gate.
                const bool host_needed = warmup || (!verify && !empty);
                EXPECT_EQ(attention.kv_cache_kernel_block_id_host.defined(), host_needed);
                if (host_needed) {
                    EXPECT_TRUE(torch::equal(attention.kv_cache_kernel_block_id_host,
                                            input.kv_cache_kernel_block_id[0].to(torch::kInt32)));
                }
            }
        }
    }
}

TEST_F(PrefillHostBlockTableTest, HostTpBroadcastChild) {
    const char* rank_env = std::getenv("RTP_TEST_HOST_TABLE_RANK");
    if (!rank_env) {
        GTEST_SKIP() << "Executed by HostTpBroadcastMatchesRoot";
    }
    const int rank = std::stoi(rank_env);
    CpuTpBroadcaster::instance().initialize(rank, 2, std::getenv("RTP_TEST_HOST_TABLE_UDS"));
    ParallelismConfig parallel;
    parallel.tp_size = 2;
    parallel.tp_rank = rank;
    for (int step = 0; step < 8; ++step) {
        const int groups = step % 2 + 1;
        const int batch = step % 3 + 1;
        const int blocks = step + 1;
        const auto expected = (torch::arange(groups * batch * blocks, torch::kInt32) + step * 100)
                                  .reshape({groups, batch, blocks});
        GptModelInputs input;
        if (rank == 0) {
            input.input_lengths = torch::full({batch}, 3, torch::kInt32).pin_memory();
            input.combo_tokens = torch::ones({batch * 3}, torch::kInt32).pin_memory();
            input.prefix_lengths = torch::zeros({batch}, torch::kInt32).pin_memory();
            input.sequence_lengths = torch::empty({0}, torch::kInt32).pin_memory();
            input.kv_cache_kernel_block_id = expected.pin_memory();
            input.kv_cache_block_id = (expected + 2000).pin_memory();
        }
        tpSyncModelInputs(input, parallel);
        EXPECT_TRUE(input.kv_cache_kernel_block_id.is_cpu());
        EXPECT_TRUE(input.kv_cache_kernel_block_id.is_pinned());
        EXPECT_TRUE(torch::equal(input.kv_cache_kernel_block_id, expected));
        EXPECT_TRUE(torch::equal(input.kv_cache_block_id, expected + 2000));
    }
    CpuTpBroadcaster::instance().reset();
}

TEST_F(PrefillHostBlockTableTest, HostTpBroadcastMatchesRoot) {
    char pattern[] = "/tmp/prefill_host_table.XXXXXX";
    const char* dir = mkdtemp(pattern);
    ASSERT_NE(dir, nullptr);
    const std::string base = std::string(dir) + "/tp";
    std::vector<pid_t> children;
    for (int rank = 0; rank < 2; ++rank) {
        std::vector<std::string> environment;
        for (char** p = environ; *p; ++p) {
            const std::string value(*p);
            // Child gtests must not overwrite/remove the parent's Bazel reports.
            if (value.find("TEST_PREMATURE_EXIT_FILE=") != 0 && value.find("XML_OUTPUT_FILE=") != 0
                && value.find("GTEST_OUTPUT=") != 0) {
                environment.push_back(value);
            }
        }
        environment.push_back("RTP_TEST_HOST_TABLE_RANK=" + std::to_string(rank));
        environment.push_back("RTP_TEST_HOST_TABLE_UDS=" + base);
        std::vector<char*> envp;
        for (auto& value : environment) {
            envp.push_back(value.data());
        }
        envp.push_back(nullptr);
        char executable[] = "/proc/self/exe";
        char filter[] = "--gtest_filter=PrefillHostBlockTableTest.HostTpBroadcastChild";
        char* args[] = {executable, filter, nullptr};
        pid_t child = -1;
        const int rc = posix_spawn(&child, executable, nullptr, nullptr, args, envp.data());
        EXPECT_EQ(rc, 0);
        if (rc == 0) {
            children.push_back(child);
        }
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (pid_t child : children) {
        int status = 0;
        pid_t result;
        do {
            result = waitpid(child, &status, WNOHANG);
            if (result == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        } while (result == 0 && std::chrono::steady_clock::now() < deadline);
        if (result == 0) {
            kill(child, SIGKILL);
            waitpid(child, &status, 0);
        }
        EXPECT_EQ(result, child) << "TP child exceeded its deadline";
        EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0) << "status=" << status;
    }
    unlink((base + "_0.sock").c_str());
    rmdir(dir);
}

TEST_F(PrefillHostBlockTableTest, MicrobatchSlicesKeepHostAndDeviceRowsAligned) {
    auto runner = model();
    GptModelInputs input;
    input.combo_tokens = torch::arange(10, torch::kInt32);
    input.input_lengths = torch::tensor({1, 2, 3, 4}, torch::kInt32);
    input.sequence_lengths = torch::empty({0}, torch::kInt32);
    input.prefix_lengths = torch::zeros({4}, torch::kInt32);
    input.lm_output_indexes = torch::tensor({0, 2, 5, 9}, torch::kInt32);
    input.kv_cache_kernel_block_id = torch::arange(24, torch::kInt32).reshape({2, 4, 3}).pin_memory();
    input.kv_cache_block_id = torch::arange(16, torch::kInt32).reshape({2, 4, 2}).pin_memory();
    MicroBatchPlan plan;
    plan.enable = true;
    plan.batch_infos = {{2, 0}, {2, 0}};
    auto split = runner->splitInputsIntoMicroBatches(input, plan);
    ASSERT_EQ(split.first.size(), 2);
    for (int batch = 0; batch < 2; ++batch) {
        const auto expected = input.kv_cache_kernel_block_id.narrow(1, batch * 2, 2);
        EXPECT_TRUE(split.first[batch].kv_cache_kernel_block_id.is_cpu());
        EXPECT_TRUE(torch::equal(split.first[batch].kv_cache_kernel_block_id, expected));
        torch_ext::PyAttentionInputs attention;
        runner->setupKVCacheForAttentionInputs(attention, split.first[batch]);
        flush(*runner);
        for (int group = 0; group < 2; ++group) {
            EXPECT_TRUE(torch::equal(attention.kv_cache_kernel_block_id_device_by_group[group].cpu(), expected[group]));
        }
        EXPECT_TRUE(torch::equal(attention.kv_cache_kernel_block_id_host, expected[0]));
    }
}

}  // namespace
}  // namespace rtp_llm
