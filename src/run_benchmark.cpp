#include "oblisc/oblisc.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

std::string require_value(int &i, int argc, char **argv) {
  if (i + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + argv[i]);
  }
  ++i;
  return argv[i];
}

std::size_t parse_size(const std::string &value) {
  if (value.empty() || value.front() == '-')
    throw std::runtime_error("expected a nonnegative integer: " + value);
  std::size_t end = 0;
  const auto parsed = std::stoull(value, &end);
  if (end != value.size() || parsed > std::numeric_limits<std::size_t>::max())
    throw std::runtime_error("invalid size: " + value);
  return static_cast<std::size_t>(parsed);
}

int parse_int(const std::string &value) {
  std::size_t end = 0;
  const auto result = std::stoi(value, &end);
  if (end != value.size()) throw std::runtime_error("invalid integer: " + value);
  return result;
}

double parse_double(const std::string &value) {
  std::size_t end = 0;
  const auto result = std::stod(value, &end);
  if (end != value.size() || !std::isfinite(result))
    throw std::runtime_error("invalid number: " + value);
  return result;
}

void print_usage() {
  std::cerr
      << "Usage: ppsc_fo[_par] --dataset-path PATH --query-path PATH "
         "--groundtruth-path PATH --dataset-size N --query-size Q --dim D "
         "--k K [--bucket-size B] [--threads T] [--output-path PATH] "
         "[--audit-path PATH] --suco-index-path PATH "
         "[--suco-index-ms MS] [--subspace-dimensionality D] "
         "[--subspace-num N] [--kmeans-num-centroid C] "
         "[--kmeans-num-iters I] [--padded-bucket-capacity C] "
         "[--path-oram true|false] [--oram-bucket-size Z] "
         "[--score-oram-blocks N] "
         "[--oblivious-sort true|false] [--use-srht true|false] "
         "[--alpha A] [--beta B] [--sc-budget-policy POLICY] "
         "[--score-selection-policy POLICY] "
         "[--sc-floor-ratio R] [--sc-quantile Q] [--sc-fixed-cells H] "
         "[--workers-per-subspace P] [--query-groups G]\n"
      << "This source release runs direct-memory diagnostics only.\n";
}

bool parse_bool(const std::string &value) {
  if (value == "1" || value == "true" || value == "True" || value == "yes" || value == "on")
    return true;
  if (value == "0" || value == "false" || value == "False" || value == "no" || value == "off")
    return false;
  throw std::runtime_error("invalid boolean: " + value);
}

}  // namespace

int main(int argc, char **argv) {
  ppsc::BenchmarkConfig config;
  config.threads = 1;
  if (const char *env = std::getenv("PPSC_NUM_THREADS")) {
    config.threads = std::max(1, parse_int(env));
  }

  try {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--dataset-path") {
        config.dataset_path = require_value(i, argc, argv);
      } else if (arg == "--query-path") {
        config.query_path = require_value(i, argc, argv);
      } else if (arg == "--groundtruth-path") {
        config.groundtruth_path = require_value(i, argc, argv);
      } else if (arg == "--output-path") {
        config.output_path = require_value(i, argc, argv);
      } else if (arg == "--audit-path") {
        config.audit_path = require_value(i, argc, argv);
      } else if (arg == "--suco-index-path") {
        config.suco_index_path = require_value(i, argc, argv);
      } else if (arg == "--suco-index-ms" || arg == "--upstream-index-ms") {
        config.upstream_index_ms = parse_double(require_value(i, argc, argv));
      } else if (arg == "--dataset-size") {
        config.dataset_size = parse_size(require_value(i, argc, argv));
      } else if (arg == "--query-size") {
        config.query_size = parse_size(require_value(i, argc, argv));
      } else if (arg == "--dim") {
        config.dim = parse_size(require_value(i, argc, argv));
      } else if (arg == "--k") {
        config.k = parse_size(require_value(i, argc, argv));
      } else if (arg == "--bucket-size") {
        config.bucket_size = parse_size(require_value(i, argc, argv));
      } else if (arg == "--subspace-dimensionality") {
        config.subspace_dimensionality = parse_size(require_value(i, argc, argv));
      } else if (arg == "--subspace-num") {
        config.subspace_num = parse_size(require_value(i, argc, argv));
      } else if (arg == "--kmeans-num-centroid") {
        config.kmeans_num_centroid = parse_size(require_value(i, argc, argv));
      } else if (arg == "--kmeans-num-iters") {
        config.kmeans_num_iters = parse_size(require_value(i, argc, argv));
      } else if (arg == "--padded-bucket-capacity") {
        config.padded_bucket_capacity = parse_size(require_value(i, argc, argv));
      } else if (arg == "--path-oram") {
        config.use_path_oram = parse_bool(require_value(i, argc, argv));
      } else if (arg == "--oram-bucket-size") {
        config.oram_bucket_size = parse_size(require_value(i, argc, argv));
      } else if (arg == "--score-oram-blocks") {
        config.score_oram_blocks = parse_size(require_value(i, argc, argv));
      } else if (arg == "--oblivious-sort") {
        config.use_oblivious_sort = parse_bool(require_value(i, argc, argv));
      } else if (arg == "--use-srht") {
        config.use_srht = parse_bool(require_value(i, argc, argv));
      } else if (arg == "--alpha") {
        config.alpha = parse_double(require_value(i, argc, argv));
      } else if (arg == "--beta") {
        config.beta = parse_double(require_value(i, argc, argv));
      } else if (arg == "--sc-budget-policy") {
        config.sc_budget_policy = require_value(i, argc, argv);
      } else if (arg == "--score-selection-policy") {
        config.score_selection_policy = require_value(i, argc, argv);
      } else if (arg == "--sc-floor-ratio") {
        config.sc_floor_ratio = parse_double(require_value(i, argc, argv));
      } else if (arg == "--sc-quantile") {
        config.sc_quantile = parse_double(require_value(i, argc, argv));
      } else if (arg == "--sc-fixed-cells") {
        config.sc_fixed_cells = parse_size(require_value(i, argc, argv));
      } else if (arg == "--threads") {
        config.threads = std::max(1, parse_int(require_value(i, argc, argv)));
      } else if (arg == "--workers-per-subspace") {
        config.workers_per_subspace = parse_int(require_value(i, argc, argv));
      } else if (arg == "--query-groups") {
        config.query_groups = parse_int(require_value(i, argc, argv));
      } else if (arg == "--help" || arg == "-h") {
        print_usage();
        return 0;
      } else {
        throw std::runtime_error("unknown argument: " + arg);
      }
    }

    if (config.dataset_path.empty() || config.query_path.empty() || config.suco_index_path.empty()) {
      print_usage();
      return 2;
    }

    if (config.threads > 100) {
      config.threads = 100;
    }
    if (config.use_path_oram)
      throw std::runtime_error("this release supports direct memory only; use --path-oram false");
    if (config.dataset_size == 0 || config.query_size == 0 || config.dim == 0 ||
        config.k == 0 || config.k > config.dataset_size || config.subspace_num == 0)
      throw std::runtime_error("positive sizes and 1 <= k <= dataset-size are required");
    if (config.dataset_size > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
      throw std::runtime_error("dataset-size exceeds int32 index identifiers");
    if (config.alpha <= 0 || config.alpha > 1 || config.beta <= 0 || config.beta > 1)
      throw std::runtime_error("alpha and beta must be in (0,1]");
    if (config.kmeans_num_centroid == 0 || config.kmeans_num_centroid > config.dataset_size)
      throw std::runtime_error("invalid centroid count");
    if (config.padded_bucket_capacity > std::numeric_limits<std::uint32_t>::max())
      throw std::runtime_error("padded capacity exceeds uint32 width");
    if (config.query_groups < 0)
      throw std::runtime_error("query-groups must be nonnegative (0 selects the maximum)");
#ifdef PPSC_PARALLEL_EXECUTABLE
    std::cout << "PPSC variant: FO-par; backend: direct; privacy_verified=false\n";
#else
    if (config.query_groups != 0)
      throw std::runtime_error("query-groups is an FO-par option");
    config.workers_per_subspace = 0;
    std::cout << "PPSC variant: FO; backend: direct; privacy_verified=false\n";
#endif

    ppsc::BenchmarkResult result = ppsc::algorithm1_run_ppsc(config);
    std::cout << "PPSC index time: " << result.index_ms << " ms\n";
    std::cout << "PPSC upstream SuCo index time: " << result.upstream_index_ms
              << " ms\n";
    std::cout << "PPSC padding materialization time: "
              << result.padding_materialization_ms << " ms\n";
    std::cout << "PPSC average query time: " << result.average_query_ms
              << " ms/query\n";
    std::cout << "PPSC average query latency: "
              << result.average_query_latency_ms << " ms/query\n";
    std::cout << "PPSC recall@" << config.k << ": " << result.recall << '\n';
    std::cout << "PPSC ORAM init time: " << result.audit.oram_init_ms
              << " ms\n";
    std::cout << "PPSC Algorithm 3 time/query: "
              << result.audit.algorithm3_ms << " ms\n";
    std::cout << "PPSC candidate projection time/query: "
              << result.audit.candidate_projection_ms << " ms\n";
    std::cout << "PPSC candidate cell sort time/query: "
              << result.audit.candidate_cell_sort_ms << " ms\n";
    std::cout << "PPSC candidate bucket scan time/query: "
              << result.audit.candidate_bucket_scan_ms << " ms\n";
    std::cout << "PPSC candidate score sort time/query: "
              << result.audit.candidate_score_sort_ms << " ms\n";
    std::cout << "PPSC Algorithm 4 time/query: "
              << result.audit.algorithm4_ms << " ms\n";
    std::cout << "PPSC refinement scan time/query: "
              << result.audit.refinement_scan_ms << " ms\n";
    std::cout << "PPSC refinement top-k sort time/query: "
              << result.audit.refinement_topk_sort_ms << " ms\n";
    std::cout << "PPSC public bucket count: " << result.public_bucket_count
              << '\n';
    std::cout << "PPSC padded bucket capacity: "
              << result.padded_bucket_capacity << '\n';
    std::cout << "PPSC padded index slots: " << result.padded_index_slots
              << '\n';
    std::cout << "PPSC real index entries: " << result.real_index_entries
              << '\n';
    std::cout << "PPSC dummy index entries: " << result.dummy_index_entries
              << '\n';
    std::cout << "PPSC truncated index entries: "
              << result.truncated_index_entries << '\n';
    std::cout << "PPSC fixed bucket count: " << result.audit.fixed_bucket_count
              << '\n';
    std::cout << "PPSC fixed candidate slots: "
              << result.audit.fixed_candidate_slots << '\n';
    std::cout << "PPSC fixed refinement slots: "
              << result.audit.fixed_refinement_slots << '\n';
    std::cout << "PPSC total vector reads: " << result.audit.total_vector_reads
              << '\n';
    std::cout << "PPSC Path-ORAM enabled: "
              << (config.use_path_oram ? "true" : "false") << '\n';
    std::cout << "PPSC ORAM logical reads/query: "
              << result.audit.oram_logical_reads << '\n';
    std::cout << "PPSC ORAM path length: " << result.audit.oram_path_length
              << '\n';
    std::cout << "PPSC ORAM bucket size: " << result.audit.oram_bucket_size
              << '\n';
    std::cout << "PPSC ORAM physical bucket reads/query: "
              << result.audit.oram_physical_bucket_reads << '\n';
    std::cout << "PPSC ORAM physical bucket writes/query: "
              << result.audit.oram_physical_bucket_writes << '\n';
    std::cout << "PPSC bucket ORAM executed: "
              << (result.audit.bucket_oram_executed ? "true" : "false")
              << '\n';
    std::cout << "PPSC score ORAM executed: "
              << (result.audit.score_oram_executed ? "true" : "false")
              << '\n';
    std::cout << "PPSC item ORAM executed: "
              << (result.audit.item_oram_executed ? "true" : "false")
              << '\n';
    std::cout << "PPSC bucket ORAM logical accesses/query: "
              << result.audit.bucket_oram_logical_accesses << '\n';
    std::cout << "PPSC score ORAM logical accesses/query: "
              << result.audit.score_oram_logical_accesses << '\n';
    std::cout << "PPSC item ORAM logical accesses/query: "
              << result.audit.item_oram_logical_accesses << '\n';
    std::cout << "PPSC ORAM stash peak: " << result.audit.oram_stash_peak
              << '\n';
    std::cout << "PPSC oblivious sort enabled: "
              << (config.use_oblivious_sort ? "true" : "false") << '\n';
    std::cout << "PPSC SRHT enabled: "
              << (config.use_srht ? "true" : "false") << '\n';
    std::cout << "PPSC SC budget policy: " << config.sc_budget_policy << '\n';
    std::cout << "PPSC SC floor ratio: " << config.sc_floor_ratio << '\n';
    std::cout << "PPSC SC quantile: " << config.sc_quantile << '\n';
    std::cout << "PPSC SC fixed cells: " << config.sc_fixed_cells << '\n';
    std::cout << "PPSC runtime hostname: "
              << result.runtime_environment.hostname << '\n';
    std::cout << "PPSC AMD SEV supported: "
              << (result.runtime_environment.amd_sev_supported ? "true" : "false")
              << '\n';
    std::cout << "PPSC AMD SEV-ES supported: "
              << (result.runtime_environment.amd_sev_es_supported ? "true" : "false")
              << '\n';
    std::cout << "PPSC /dev/sev present: "
              << (result.runtime_environment.dev_sev_present ? "true" : "false")
              << '\n';
    std::cout << "PPSC /dev/sev readable: "
              << (result.runtime_environment.dev_sev_readable ? "true" : "false")
              << '\n';
    std::cout << "PPSC /dev/kvm present: "
              << (result.runtime_environment.dev_kvm_present ? "true" : "false")
              << '\n';
    std::cout << "PPSC /dev/kvm readable: "
              << (result.runtime_environment.dev_kvm_readable ? "true" : "false")
              << '\n';
    std::cout << "PPSC /dev/kvm writable: "
              << (result.runtime_environment.dev_kvm_writable ? "true" : "false")
              << '\n';
    std::cout << "PPSC cell oblivious sort executed: "
              << (result.audit.cell_oblivious_sort_executed ? "true" : "false")
              << '\n';
    std::cout << "PPSC cell sort input size: "
              << result.audit.cell_sort_input_size << '\n';
    std::cout << "PPSC cell sort padded size: "
              << result.audit.cell_sort_padded_size << '\n';
    std::cout << "PPSC cell sort compare-exchanges: "
              << result.audit.cell_sort_compare_exchanges << '\n';
    std::cout << "PPSC score oblivious sort executed: "
              << (result.audit.score_oblivious_sort_executed ? "true" : "false")
              << '\n';
    std::cout << "PPSC score sort input size: "
              << result.audit.score_sort_input_size << '\n';
    std::cout << "PPSC score sort padded size: "
              << result.audit.score_sort_padded_size << '\n';
    std::cout << "PPSC score sort compare-exchanges: "
              << result.audit.score_sort_compare_exchanges << '\n';
    std::cout << "PPSC top-k oblivious sort executed: "
              << (result.audit.topk_oblivious_sort_executed ? "true" : "false")
              << '\n';
    std::cout << "PPSC oblivious sort input size: "
              << result.audit.oblivious_sort_input_size << '\n';
    std::cout << "PPSC oblivious sort padded size: "
              << result.audit.oblivious_sort_padded_size << '\n';
    std::cout << "PPSC oblivious sort compare-exchanges: "
              << result.audit.oblivious_sort_compare_exchanges << '\n';
    std::cout << "PPSC query-independent shape: "
              << (result.audit.query_independent_shape ? "true" : "false")
              << '\n';
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
