#include "oblisc/oblisc.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ppsc {

namespace {

std::size_t ceil_to_size(double value) {
  if (value <= 0.0) {
    return 0;
  }
  return static_cast<std::size_t>(std::ceil(value));
}

bool uses_buffer_score_selection(const BenchmarkConfig &config) {
  return config.score_selection_policy == "heap" ||
         config.score_selection_policy == "topr_heap" ||
         config.score_selection_policy == "topr-heap" ||
         config.score_selection_policy == "selection";
}

std::size_t score_oram_block_count(const BenchmarkConfig &config) {
  if (config.score_oram_blocks > 0) {
    return config.score_oram_blocks;
  }
  if (uses_buffer_score_selection(config)) {
    return std::max<std::size_t>(
        config.k, ceil_to_size(config.beta * static_cast<double>(config.dataset_size)));
  }
  return config.dataset_size;
}

}  // namespace

BenchmarkResult algorithm1_run_ppsc(const BenchmarkConfig &config) {
  if (config.dataset_size == 0 || config.query_size == 0 || config.dim == 0 ||
      config.k == 0) {
    throw std::runtime_error("dataset_size, query_size, dim, and k must be positive");
  }

  std::cerr << "Loading dataset: " << config.dataset_path << '\n';
  Matrix dataset =
      load_raw_f32(config.dataset_path, config.dataset_size, config.dim);
  std::cerr << "Loading queries: " << config.query_path << '\n';
  Matrix queries = load_raw_f32(config.query_path, config.query_size, config.dim);

  const auto index_start = std::chrono::steady_clock::now();
  PublicPaddedIndex index = algorithm2_build_public_padded_index(config);
  const auto index_end = std::chrono::steady_clock::now();

  int worker_count =
      std::max(1, std::min(config.threads,
                           static_cast<int>(config.query_size)));
#ifndef _OPENMP
  worker_count = 1;
#else
  omp_set_num_threads(worker_count);
#endif

  std::vector<PrivateExecutorContext> contexts(
      static_cast<std::size_t>(worker_count));
  double oram_init_ms = 0.0;
  if (config.use_path_oram) {
    const std::size_t score_blocks = score_oram_block_count(config);
    std::cerr << "Initializing Path ORAM backends for buckets, scores, and items "
              << "for " << worker_count << " workers.\n";
    const auto oram_init_start = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static)
    for (int worker = 0; worker < worker_count; ++worker) {
      PrivateExecutorContext &context =
          contexts[static_cast<std::size_t>(worker)];
      context.path_oram_enabled = true;
      const std::uint64_t seed_delta =
          static_cast<std::uint64_t>(worker) * 0x9e3779b97f4a7c15ULL;
      context.bucket_oram.reset(index.public_bucket_count,
                                config.oram_bucket_size,
                                0x505053435f42554bULL ^ seed_delta);
      context.score_oram.reset(score_blocks, config.oram_bucket_size,
                               0x505053435f53434fULL ^ seed_delta);
      const std::size_t item_blocks =
          uses_buffer_score_selection(config) ? score_blocks : index.dataset_size;
      context.item_oram.reset(item_blocks, config.oram_bucket_size,
                              0x505053435f49544dULL ^ seed_delta);
    }
    const auto oram_init_end = std::chrono::steady_clock::now();
    oram_init_ms =
        std::chrono::duration<double, std::milli>(oram_init_end -
                                                  oram_init_start)
            .count();
  } else {
    for (PrivateExecutorContext &context : contexts) {
      context.path_oram_enabled = false;
    }
  }

  std::vector<QueryResult> query_results(config.query_size);
  const auto query_batch_start = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static)
  for (std::int64_t query_index = 0;
       query_index < static_cast<std::int64_t>(config.query_size);
       ++query_index) {
    const std::size_t i = static_cast<std::size_t>(query_index);
#ifdef _OPENMP
    const int worker = omp_get_thread_num();
#else
    const int worker = 0;
#endif
    PrivateExecutorContext &context =
        contexts[static_cast<std::size_t>(worker)];
    const auto query_start = std::chrono::steady_clock::now();
    AccessAudit activation_audit;
    std::vector<std::int64_t> slots =
        algorithm3_fo_candidate_activation(index, queries.row(i), config,
                                           context, activation_audit);
    QueryResult query_result = algorithm4_fo_topk_refinement(
        dataset, queries.row(i), slots, index, config, context);
    const auto query_end = std::chrono::steady_clock::now();
    query_result.query_ms =
        std::chrono::duration<double, std::milli>(query_end - query_start).count();
    query_result.audit.fixed_bucket_count = activation_audit.fixed_bucket_count;
    query_result.audit.fixed_candidate_slots =
        activation_audit.fixed_candidate_slots;
    query_result.audit.cell_sort_input_size = activation_audit.cell_sort_input_size;
    query_result.audit.cell_sort_padded_size = activation_audit.cell_sort_padded_size;
    query_result.audit.cell_sort_compare_exchanges =
        activation_audit.cell_sort_compare_exchanges;
    query_result.audit.score_sort_input_size = activation_audit.score_sort_input_size;
    query_result.audit.score_sort_padded_size =
        activation_audit.score_sort_padded_size;
    query_result.audit.score_sort_compare_exchanges =
        activation_audit.score_sort_compare_exchanges;
    query_result.audit.bucket_oram_logical_accesses =
        activation_audit.bucket_oram_logical_accesses;
    query_result.audit.score_oram_logical_accesses =
        activation_audit.score_oram_logical_accesses;
    query_result.audit.bucket_oram_executed =
        activation_audit.bucket_oram_executed;
    query_result.audit.score_oram_executed =
        activation_audit.score_oram_executed;
    query_result.audit.cell_oblivious_sort_executed =
        activation_audit.cell_oblivious_sort_executed;
    query_result.audit.score_oblivious_sort_executed =
        activation_audit.score_oblivious_sort_executed;
    query_result.audit.oram_stash_peak =
        std::max(query_result.audit.oram_stash_peak,
                 activation_audit.oram_stash_peak);
    query_result.audit.oram_logical_reads =
        query_result.audit.bucket_oram_logical_accesses +
        query_result.audit.score_oram_logical_accesses +
        query_result.audit.item_oram_logical_accesses;
    query_result.audit.oram_path_length =
        context.path_oram_enabled ? context.item_oram.stats().height + 1 : 0;
    query_result.audit.oram_bucket_size =
        context.path_oram_enabled ? config.oram_bucket_size : 0;
    query_result.audit.oram_physical_bucket_reads =
        query_result.audit.oram_logical_reads *
        query_result.audit.oram_path_length * query_result.audit.oram_bucket_size;
    query_result.audit.oram_physical_bucket_writes =
        query_result.audit.oram_physical_bucket_reads;
    query_result.audit.query_independent_shape =
        query_result.audit.query_independent_shape &&
        activation_audit.query_independent_shape;
    query_results[i] = std::move(query_result);
    if (i % 10 == 0) {
#pragma omp critical
      std::cerr << "Processed query " << i << '\n';
    }
  }
  const auto query_batch_end = std::chrono::steady_clock::now();

  BenchmarkResult result;
  const double measured_index_ms =
      std::chrono::duration<double, std::milli>(index_end - index_start).count();
  result.upstream_index_ms = index.upstream_index_ms;
  result.padding_materialization_ms =
      index.padding_materialization_ms > 0.0 ? index.padding_materialization_ms
                                             : measured_index_ms;
  result.index_ms = result.upstream_index_ms + result.padding_materialization_ms;
  const double query_batch_ms =
      std::chrono::duration<double, std::milli>(query_batch_end -
                                                query_batch_start)
          .count();
  double total_query_latency_ms = 0.0;
  for (const QueryResult &query_result : query_results) {
    total_query_latency_ms += query_result.query_ms;
  }
  result.average_query_ms =
      query_batch_ms / static_cast<double>(config.query_size);
  result.average_query_latency_ms =
      total_query_latency_ms / static_cast<double>(config.query_size);
  result.audit = query_results.empty() ? AccessAudit{} : query_results.front().audit;
  if (!query_results.empty()) {
    double algorithm3_ms = 0.0;
    double projection_ms = 0.0;
    double cell_sort_ms = 0.0;
    double bucket_scan_ms = 0.0;
    double score_sort_ms = 0.0;
    double algorithm4_ms = 0.0;
    double refinement_scan_ms = 0.0;
    double refinement_topk_sort_ms = 0.0;
    for (const QueryResult &query_result : query_results) {
      algorithm3_ms += query_result.audit.algorithm3_ms;
      projection_ms += query_result.audit.candidate_projection_ms;
      cell_sort_ms += query_result.audit.candidate_cell_sort_ms;
      bucket_scan_ms += query_result.audit.candidate_bucket_scan_ms;
      score_sort_ms += query_result.audit.candidate_score_sort_ms;
      algorithm4_ms += query_result.audit.algorithm4_ms;
      refinement_scan_ms += query_result.audit.refinement_scan_ms;
      refinement_topk_sort_ms += query_result.audit.refinement_topk_sort_ms;
    }
    const double denom = static_cast<double>(query_results.size());
    result.audit.algorithm3_ms = algorithm3_ms / denom;
    result.audit.candidate_projection_ms = projection_ms / denom;
    result.audit.candidate_cell_sort_ms = cell_sort_ms / denom;
    result.audit.candidate_bucket_scan_ms = bucket_scan_ms / denom;
    result.audit.candidate_score_sort_ms = score_sort_ms / denom;
    result.audit.algorithm4_ms = algorithm4_ms / denom;
    result.audit.refinement_scan_ms = refinement_scan_ms / denom;
    result.audit.refinement_topk_sort_ms = refinement_topk_sort_ms / denom;
  }
  result.audit.oram_init_ms = oram_init_ms;
  result.audit.total_vector_reads = result.audit.fixed_refinement_slots *
                                    static_cast<std::size_t>(config.query_size);
  result.public_bucket_count = index.public_bucket_count;
  result.padded_bucket_capacity = index.padded_bucket_capacity;
  result.padded_index_slots = index.padded_index_slots;
  result.real_index_entries = index.real_index_entries;
  result.dummy_index_entries = index.dummy_index_entries;
  result.truncated_index_entries = index.truncated_index_entries;
  result.subspace_min_cell_sizes = index.subspace_min_cell_sizes;
  result.subspace_max_cell_sizes = index.subspace_max_cell_sizes;
  result.runtime_environment = inspect_runtime_environment();

  if (!config.groundtruth_path.empty()) {
    GroundTruth truth =
        load_raw_i64_groundtruth(config.groundtruth_path, config.query_size, config.k);
    result.recall = recall_at_k(truth, query_results, config.k);
  }

  save_results(config.output_path, query_results, config.k);
  save_audit(config.audit_path, config, result);
  return result;
}

}  // namespace ppsc
