#include "ppsc/ppsc.hpp"

#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace ppsc {

namespace {

template <typename T>
void read_exact(const std::string &path, std::vector<T> &buffer) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open file: " + path);
  }
  input.read(reinterpret_cast<char *>(buffer.data()),
             static_cast<std::streamsize>(buffer.size() * sizeof(T)));
  if (!input) {
    throw std::runtime_error("short read from file: " + path);
  }
}

bool file_contains_one(const std::string &path) {
  std::ifstream input(path);
  std::string value;
  input >> value;
  return value == "1" || value == "Y" || value == "y";
}

bool path_exists(const std::string &path) {
  return ::access(path.c_str(), F_OK) == 0;
}

bool path_readable(const std::string &path) {
  return ::access(path.c_str(), R_OK) == 0;
}

bool path_writable(const std::string &path) {
  return ::access(path.c_str(), W_OK) == 0;
}

}  // namespace

Matrix load_raw_f32(const std::string &path, std::size_t rows, std::size_t cols) {
  Matrix matrix;
  matrix.rows = rows;
  matrix.cols = cols;
  matrix.values.resize(rows * cols);
  read_exact(path, matrix.values);
  return matrix;
}

GroundTruth load_raw_i64_groundtruth(const std::string &path, std::size_t rows,
                                     std::size_t k) {
  GroundTruth truth;
  truth.rows = rows;
  truth.k = k;
  truth.ids.resize(rows * k);
  read_exact(path, truth.ids);
  return truth;
}

void save_results(const std::string &path, const std::vector<QueryResult> &results,
                  std::size_t k) {
  if (path.empty()) {
    return;
  }
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("cannot open output file: " + path);
  }
  for (const auto &result : results) {
    for (std::size_t j = 0; j < k; ++j) {
      if (j != 0) {
        output << ' ';
      }
      output << result.ids[j];
    }
    output << '\n';
  }
}

void save_audit(const std::string &path, const BenchmarkConfig &config,
                const BenchmarkResult &result) {
  if (path.empty()) {
    return;
  }
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("cannot open audit file: " + path);
  }
  output << "variant=" << (config.workers_per_subspace > 0 ? "FO-par" : "FO") << '\n';
  output << "memory_backend=direct\naccess_pattern_protection=false\nprivacy_verified=false\n";
  output << "workers_per_subspace=" << config.workers_per_subspace << '\n';
  output << "intra_query_threads=" << config.intra_query_threads << '\n';
  output << "query_groups_arg=" << config.query_groups << '\n';
  output << "legacy_counters_are_not_protected_payload_accesses=true\n";
  output << "dataset_size=" << config.dataset_size << '\n';
  output << "query_size=" << config.query_size << '\n';
  output << "dimension=" << config.dim << '\n';
  output << "k=" << config.k << '\n';
  output << "bucket_size=" << config.bucket_size << '\n';
  output << "subspace_dimensionality=" << config.subspace_dimensionality << '\n';
  output << "subspace_num=" << config.subspace_num << '\n';
  output << "kmeans_num_centroid=" << config.kmeans_num_centroid << '\n';
  output << "kmeans_num_iters=" << config.kmeans_num_iters << '\n';
  output << "padded_bucket_capacity_arg=" << config.padded_bucket_capacity << '\n';
  output << "path_oram=" << (config.use_path_oram ? "true" : "false") << '\n';
  const char *oram_threshold = std::getenv("PPSC_ORAM_LIGHTWEIGHT_THRESHOLD");
  output << "oram_lightweight_threshold="
         << (oram_threshold == nullptr || *oram_threshold == '\0'
                 ? "100000"
                 : oram_threshold)
         << '\n';
  output << "oram_bucket_size_arg=" << config.oram_bucket_size << '\n';
  output << "score_oram_blocks_arg=" << config.score_oram_blocks << '\n';
  output << "oblivious_sort="
         << (config.use_oblivious_sort ? "true" : "false") << '\n';
  output << "use_srht=" << (config.use_srht ? "true" : "false") << '\n';
  output << "alpha=" << config.alpha << '\n';
  output << "beta=" << config.beta << '\n';
  output << "sc_budget_policy=" << config.sc_budget_policy << '\n';
  output << "score_selection_policy=" << config.score_selection_policy << '\n';
  output << "sc_floor_ratio=" << config.sc_floor_ratio << '\n';
  output << "sc_quantile=" << config.sc_quantile << '\n';
  output << "sc_fixed_cells=" << config.sc_fixed_cells << '\n';
  output << "suco_index_path=" << config.suco_index_path << '\n';
  output << "threads=" << config.threads << '\n';
  output << "runtime_hostname=" << result.runtime_environment.hostname << '\n';
  output << "amd_sev_supported="
         << (result.runtime_environment.amd_sev_supported ? "true" : "false")
         << '\n';
  output << "amd_sev_es_supported="
         << (result.runtime_environment.amd_sev_es_supported ? "true" : "false")
         << '\n';
  output << "dev_sev_present="
         << (result.runtime_environment.dev_sev_present ? "true" : "false")
         << '\n';
  output << "dev_sev_readable="
         << (result.runtime_environment.dev_sev_readable ? "true" : "false")
         << '\n';
  output << "dev_kvm_present="
         << (result.runtime_environment.dev_kvm_present ? "true" : "false")
         << '\n';
  output << "dev_kvm_readable="
         << (result.runtime_environment.dev_kvm_readable ? "true" : "false")
         << '\n';
  output << "dev_kvm_writable="
         << (result.runtime_environment.dev_kvm_writable ? "true" : "false")
         << '\n';
  output << "index_ms=" << result.index_ms << '\n';
  output << "upstream_index_ms=" << result.upstream_index_ms << '\n';
  output << "padding_materialization_ms=" << result.padding_materialization_ms
         << '\n';
  output << "average_query_ms=" << result.average_query_ms << '\n';
  output << "average_query_latency_ms=" << result.average_query_latency_ms
         << '\n';
  output << "recall=" << result.recall << '\n';
  output << "oram_init_ms=" << result.audit.oram_init_ms << '\n';
  output << "algorithm3_ms_per_query=" << result.audit.algorithm3_ms << '\n';
  output << "candidate_projection_ms_per_query="
         << result.audit.candidate_projection_ms << '\n';
  output << "candidate_cell_sort_ms_per_query="
         << result.audit.candidate_cell_sort_ms << '\n';
  output << "candidate_bucket_scan_ms_per_query="
         << result.audit.candidate_bucket_scan_ms << '\n';
  output << "candidate_score_sort_ms_per_query="
         << result.audit.candidate_score_sort_ms << '\n';
  output << "algorithm4_ms_per_query=" << result.audit.algorithm4_ms << '\n';
  output << "refinement_scan_ms_per_query="
         << result.audit.refinement_scan_ms << '\n';
  output << "refinement_topk_sort_ms_per_query="
         << result.audit.refinement_topk_sort_ms << '\n';
  output << "public_bucket_count=" << result.public_bucket_count << '\n';
  output << "padded_bucket_capacity=" << result.padded_bucket_capacity << '\n';
  output << "padded_index_slots=" << result.padded_index_slots << '\n';
  output << "real_index_entries=" << result.real_index_entries << '\n';
  output << "dummy_index_entries=" << result.dummy_index_entries << '\n';
  output << "truncated_index_entries=" << result.truncated_index_entries << '\n';
  output << "subspace_min_cell_sizes=";
  for (std::size_t i = 0; i < result.subspace_min_cell_sizes.size(); ++i) {
    output << (i == 0 ? "" : ",") << result.subspace_min_cell_sizes[i];
  }
  output << '\n';
  output << "subspace_max_cell_sizes=";
  for (std::size_t i = 0; i < result.subspace_max_cell_sizes.size(); ++i) {
    output << (i == 0 ? "" : ",") << result.subspace_max_cell_sizes[i];
  }
  output << '\n';
  output << "fixed_bucket_count=" << result.audit.fixed_bucket_count << '\n';
  output << "fixed_candidate_slots=" << result.audit.fixed_candidate_slots << '\n';
  output << "fixed_refinement_slots=" << result.audit.fixed_refinement_slots << '\n';
  output << "total_vector_reads=" << result.audit.total_vector_reads << '\n';
  output << "oram_logical_reads_per_query="
         << result.audit.oram_logical_reads << '\n';
  output << "oram_path_length=" << result.audit.oram_path_length << '\n';
  output << "oram_bucket_size=" << result.audit.oram_bucket_size << '\n';
  output << "oram_physical_bucket_reads_per_query="
         << result.audit.oram_physical_bucket_reads << '\n';
  output << "oram_physical_bucket_writes_per_query="
         << result.audit.oram_physical_bucket_writes << '\n';
  output << "bucket_oram_executed="
         << (result.audit.bucket_oram_executed ? "true" : "false") << '\n';
  output << "score_oram_executed="
         << (result.audit.score_oram_executed ? "true" : "false") << '\n';
  output << "item_oram_executed="
         << (result.audit.item_oram_executed ? "true" : "false") << '\n';
  output << "bucket_oram_logical_accesses_per_query="
         << result.audit.bucket_oram_logical_accesses << '\n';
  output << "score_oram_logical_accesses_per_query="
         << result.audit.score_oram_logical_accesses << '\n';
  output << "item_oram_logical_accesses_per_query="
         << result.audit.item_oram_logical_accesses << '\n';
  output << "oram_stash_peak=" << result.audit.oram_stash_peak << '\n';
  output << "cell_oblivious_sort_executed="
         << (result.audit.cell_oblivious_sort_executed ? "true" : "false")
         << '\n';
  output << "cell_sort_input_size=" << result.audit.cell_sort_input_size << '\n';
  output << "cell_sort_padded_size=" << result.audit.cell_sort_padded_size
         << '\n';
  output << "cell_sort_compare_exchanges="
         << result.audit.cell_sort_compare_exchanges << '\n';
  output << "score_oblivious_sort_executed="
         << (result.audit.score_oblivious_sort_executed ? "true" : "false")
         << '\n';
  output << "score_sort_input_size=" << result.audit.score_sort_input_size
         << '\n';
  output << "score_sort_padded_size=" << result.audit.score_sort_padded_size
         << '\n';
  output << "score_sort_compare_exchanges="
         << result.audit.score_sort_compare_exchanges << '\n';
  output << "topk_oblivious_sort_executed="
         << (result.audit.topk_oblivious_sort_executed ? "true" : "false")
         << '\n';
  output << "oblivious_sort_input_size="
         << result.audit.oblivious_sort_input_size << '\n';
  output << "oblivious_sort_padded_size="
         << result.audit.oblivious_sort_padded_size << '\n';
  output << "oblivious_sort_compare_exchanges="
         << result.audit.oblivious_sort_compare_exchanges << '\n';
  output << "query_independent_shape="
         << (result.audit.query_independent_shape ? "true" : "false") << '\n';
}

RuntimeEnvironmentAudit inspect_runtime_environment() {
  RuntimeEnvironmentAudit audit;
  char hostname[256] = {};
  if (::gethostname(hostname, sizeof(hostname) - 1) == 0) {
    audit.hostname = hostname;
  }
  audit.amd_sev_supported =
      file_contains_one("/sys/module/kvm_amd/parameters/sev");
  audit.amd_sev_es_supported =
      file_contains_one("/sys/module/kvm_amd/parameters/sev_es");
  audit.dev_sev_present = path_exists("/dev/sev");
  audit.dev_sev_readable = path_readable("/dev/sev");
  audit.dev_kvm_present = path_exists("/dev/kvm");
  audit.dev_kvm_readable = path_readable("/dev/kvm");
  audit.dev_kvm_writable = path_writable("/dev/kvm");
  return audit;
}

}  // namespace ppsc
