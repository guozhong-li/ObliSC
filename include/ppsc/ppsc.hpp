#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace ppsc {

struct BenchmarkConfig {
  std::string dataset_path;
  std::string query_path;
  std::string groundtruth_path;
  std::string output_path;
  std::string audit_path;
  std::string suco_index_path;
  std::size_t dataset_size = 0;
  std::size_t query_size = 0;
  std::size_t dim = 0;
  std::size_t k = 50;
  std::size_t bucket_size = 1024;
  std::size_t subspace_dimensionality = 0;
  std::size_t subspace_num = 8;
  std::size_t kmeans_num_centroid = 50;
  std::size_t kmeans_num_iters = 2;
  std::size_t padded_bucket_capacity = 0;
  std::size_t oram_bucket_size = 4;
  std::size_t score_oram_blocks = 0;
  bool use_path_oram = false;
  bool use_oblivious_sort = true;
  bool use_srht = true;
  double alpha = 0.03;
  double beta = 0.003;
  std::string sc_budget_policy = "fixed";
  std::string score_selection_policy = "heap";
  double sc_floor_ratio = 0.5;
  double sc_quantile = 0.5;
  std::size_t sc_fixed_cells = 100;
  double upstream_index_ms = 0.0;
  int threads = 1;
  int intra_query_threads = 1;
  int workers_per_subspace = 2;
  int query_groups = 0;
};

struct Matrix {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::vector<float> values;

  const float *row(std::size_t i) const {
    return values.data() + i * cols;
  }
};

struct GroundTruth {
  std::size_t rows = 0;
  std::size_t k = 0;
  std::vector<std::int64_t> ids;

  const std::int64_t *row(std::size_t i) const {
    return ids.data() + i * k;
  }
};

struct PublicPaddedIndex {
  std::size_t dataset_size = 0;
  std::size_t bucket_size = 0;
  std::size_t bucket_count = 0;
  std::size_t padded_slots = 0;
  std::size_t subspace_num = 0;
  std::size_t subspace_dimensionality = 0;
  std::size_t kmeans_dim = 0;
  std::size_t kmeans_num_centroid = 0;
  std::size_t cell_count = 0;
  std::size_t public_bucket_count = 0;
  std::size_t padded_bucket_capacity = 0;
  std::size_t padded_index_slots = 0;
  std::size_t real_index_entries = 0;
  std::size_t dummy_index_entries = 0;
  std::size_t truncated_index_entries = 0;
  double upstream_index_ms = 0.0;
  double padding_materialization_ms = 0.0;
  std::vector<float> centroids;
  std::vector<std::uint32_t> bucket_counts;
  std::vector<std::uint32_t> subspace_min_cell_sizes;
  std::vector<std::uint32_t> subspace_max_cell_sizes;
  std::vector<std::uint64_t> bucket_offsets;
  std::vector<std::int32_t> padded_bucket_ids;
};

struct AccessAudit {
  int scan_workers_observed = 0;
  int selection_workers_observed = 0;
  int refinement_workers_observed = 0;
  double oram_init_ms = 0.0;
  double algorithm3_ms = 0.0;
  double state_init_ms = 0.0;
  double cell_prepare_ms = 0.0;
  double scan_prepare_ms = 0.0;
  double score_reduce_ms = 0.0;
  double candidate_projection_ms = 0.0;
  double candidate_cell_sort_ms = 0.0;
  double candidate_bucket_scan_ms = 0.0;
  double candidate_score_sort_ms = 0.0;
  double algorithm4_ms = 0.0;
  double refinement_scan_ms = 0.0;
  double refinement_topk_sort_ms = 0.0;
  std::size_t fixed_bucket_count = 0;
  std::size_t fixed_candidate_slots = 0;
  std::size_t fixed_refinement_slots = 0;
  std::size_t total_vector_reads = 0;
  std::size_t oram_logical_reads = 0;
  std::size_t oram_path_length = 0;
  std::size_t oram_bucket_size = 0;
  std::size_t oram_physical_bucket_reads = 0;
  std::size_t oram_physical_bucket_writes = 0;
  std::size_t oblivious_sort_input_size = 0;
  std::size_t oblivious_sort_padded_size = 0;
  std::size_t oblivious_sort_compare_exchanges = 0;
  std::size_t cell_sort_input_size = 0;
  std::size_t cell_sort_padded_size = 0;
  std::size_t cell_sort_compare_exchanges = 0;
  std::size_t score_sort_input_size = 0;
  std::size_t score_sort_padded_size = 0;
  std::size_t score_sort_compare_exchanges = 0;
  std::size_t bucket_oram_logical_accesses = 0;
  std::size_t score_oram_logical_accesses = 0;
  std::size_t item_oram_logical_accesses = 0;
  std::size_t oram_stash_peak = 0;
  bool bucket_oram_executed = false;
  bool score_oram_executed = false;
  bool item_oram_executed = false;
  bool cell_oblivious_sort_executed = false;
  bool score_oblivious_sort_executed = false;
  bool topk_oblivious_sort_executed = false;
  bool query_independent_shape = true;
};

struct RuntimeEnvironmentAudit {
  std::string hostname;
  bool amd_sev_supported = false;
  bool amd_sev_es_supported = false;
  bool dev_sev_present = false;
  bool dev_sev_readable = false;
  bool dev_kvm_present = false;
  bool dev_kvm_readable = false;
  bool dev_kvm_writable = false;
};

struct QueryResult {
  std::vector<std::int64_t> ids;
  double query_ms = 0.0;
  AccessAudit audit;
};

struct BenchmarkResult {
  double index_ms = 0.0;
  double upstream_index_ms = 0.0;
  double padding_materialization_ms = 0.0;
  double average_query_ms = 0.0;
  double average_query_latency_ms = 0.0;
  double recall = 0.0;
  AccessAudit audit;
  std::size_t public_bucket_count = 0;
  std::size_t padded_bucket_capacity = 0;
  std::size_t padded_index_slots = 0;
  std::size_t real_index_entries = 0;
  std::size_t dummy_index_entries = 0;
  std::size_t truncated_index_entries = 0;
  std::vector<std::uint32_t> subspace_min_cell_sizes;
  std::vector<std::uint32_t> subspace_max_cell_sizes;
  RuntimeEnvironmentAudit runtime_environment;
};

struct PathOramStats {
  std::size_t logical_accesses = 0;
  std::size_t path_bucket_reads = 0;
  std::size_t path_bucket_writes = 0;
  std::size_t stash_peak = 0;
  std::size_t height = 0;
  std::size_t leaf_count = 0;
  std::size_t bucket_size = 0;
  bool executed = false;
};

class PathOram {
 public:
  PathOram() = default;
  PathOram(std::size_t logical_blocks, std::size_t bucket_size,
           std::uint64_t seed);

  void reset(std::size_t logical_blocks, std::size_t bucket_size,
             std::uint64_t seed);
  void access(std::size_t logical_block);
  void record_lightweight_accesses(std::size_t count);
  const PathOramStats &stats() const { return stats_; }
  std::size_t logical_blocks() const { return logical_blocks_; }
  bool lightweight() const { return lightweight_; }

 private:
  std::uint32_t random_leaf(std::uint64_t counter) const;
  bool block_fits_node(std::int32_t block, std::size_t node) const;
  std::size_t leaf_node(std::uint32_t leaf) const;
  void place_initial_block(std::int32_t block);
  void read_path(std::uint32_t leaf);
  void write_path(std::uint32_t leaf);

  std::size_t logical_blocks_ = 0;
  std::size_t bucket_size_ = 0;
  std::size_t leaf_count_ = 1;
  std::size_t leaf_base_ = 1;
  std::uint64_t seed_ = 0;
  std::uint64_t random_counter_ = 0;
  std::vector<std::uint32_t> position_map_;
  std::vector<std::int32_t> tree_;
  std::vector<std::int32_t> stash_;
  PathOramStats stats_;
  bool lightweight_ = false;
};

struct PrivateExecutorContext {
  PathOram bucket_oram;
  PathOram score_oram;
  PathOram item_oram;
  bool path_oram_enabled = false;
};

Matrix load_raw_f32(const std::string &path, std::size_t rows, std::size_t cols);
GroundTruth load_raw_i64_groundtruth(const std::string &path, std::size_t rows,
                                     std::size_t k);
void save_results(const std::string &path, const std::vector<QueryResult> &results,
                  std::size_t k);
void save_audit(const std::string &path, const BenchmarkConfig &config,
                const BenchmarkResult &result);
RuntimeEnvironmentAudit inspect_runtime_environment();

PublicPaddedIndex algorithm2_build_public_padded_index(
    const BenchmarkConfig &config);
std::vector<std::int64_t> algorithm3_fo_candidate_activation(
    const PublicPaddedIndex &index, const float *query,
    const BenchmarkConfig &config, PrivateExecutorContext &context,
    AccessAudit &audit);
QueryResult algorithm4_fo_topk_refinement(const Matrix &dataset,
                                          const float *query,
                                          const std::vector<std::int64_t> &slots,
                                          const PublicPaddedIndex &index,
                                          const BenchmarkConfig &config,
                                          PrivateExecutorContext &context);
BenchmarkResult algorithm1_run_ppsc(const BenchmarkConfig &config);

double recall_at_k(const GroundTruth &truth, const std::vector<QueryResult> &results,
                   std::size_t k);

}  // namespace ppsc
