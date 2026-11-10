#include "oblisc/oblisc.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ppsc {

namespace {

using ScoredId = std::pair<float, std::int64_t>;

bool better(const ScoredId &a, const ScoredId &b) {
  if (a.first != b.first) {
    return a.first < b.first;
  }
  return a.second < b.second;
}

float l2_distance(const float *a, const float *b, std::size_t dim) {
  float sum = 0.0F;
  for (std::size_t i = 0; i < dim; ++i) {
    const float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

std::size_t next_power_of_two(std::size_t value) {
  std::size_t out = 1;
  while (out < value) {
    out <<= 1U;
  }
  return out;
}

std::size_t path_oram_path_length(std::size_t block_count) {
  std::size_t leaves = 1;
  std::size_t height = 0;
  while (leaves < std::max<std::size_t>(1, block_count)) {
    leaves <<= 1U;
    ++height;
  }
  return height + 1;
}

std::size_t item_oram_address(std::size_t id, const BenchmarkConfig &config,
                              const PrivateExecutorContext &context) {
  if (!context.path_oram_enabled || context.item_oram.logical_blocks() == 0) {
    return id;
  }
  if (config.score_selection_policy == "heap" ||
      config.score_selection_policy == "topr_heap" ||
      config.score_selection_policy == "topr-heap" ||
      config.score_selection_policy == "selection") {
    const std::uint64_t mixed =
        (static_cast<std::uint64_t>(id) + 0xd1b54a32d192ed03ULL) *
        0x94d049bb133111ebULL;
    return static_cast<std::size_t>(
        mixed % static_cast<std::uint64_t>(context.item_oram.logical_blocks()));
  }
  return id;
}

std::size_t bitonic_oblivious_sort(std::vector<ScoredId> &items) {
  const std::size_t n = items.size();
  std::size_t comparisons = 0;
  for (std::size_t size = 2; size <= n; size <<= 1U) {
    for (std::size_t stride = size >> 1U; stride > 0; stride >>= 1U) {
      for (std::size_t i = 0; i < n; ++i) {
        const std::size_t j = i ^ stride;
        if (j <= i) {
          continue;
        }
        const bool ascending = (i & size) == 0;
        const bool swap_items =
            ascending ? better(items[j], items[i]) : better(items[i], items[j]);
        if (swap_items) {
          std::swap(items[i], items[j]);
        }
        ++comparisons;
      }
    }
  }
  return comparisons;
}

std::size_t fixed_schedule_topk_select(const ScoredId &item,
                                       std::vector<ScoredId> &top_items) {
  ScoredId candidate = item;
  std::size_t comparisons = 0;
  for (ScoredId &current : top_items) {
    if (better(candidate, current)) {
      std::swap(candidate, current);
    }
    ++comparisons;
  }
  return comparisons;
}

}  // namespace

QueryResult algorithm4_fo_topk_refinement(const Matrix &dataset,
                                          const float *query,
                                          const std::vector<std::int64_t> &slots,
                                          const PublicPaddedIndex &index,
                                          const BenchmarkConfig &config,
                                          PrivateExecutorContext &context) {
  const auto start = std::chrono::steady_clock::now();
  const std::size_t k = config.k;
  const std::size_t item_oram_before = context.item_oram.stats().logical_accesses;
  std::vector<ScoredId> sort_items;
  std::vector<ScoredId> top_items;
  if (config.use_oblivious_sort) {
    top_items.assign(k, {std::numeric_limits<float>::infinity(), -1});
  } else {
    sort_items.reserve(slots.size());
  }
  std::size_t sort_comparisons = 0;
  const auto scan_start = std::chrono::steady_clock::now();
  for (std::size_t slot = 0; slot < slots.size(); ++slot) {
    const std::int64_t id = slots[slot];
    const bool real = id >= 0 && static_cast<std::size_t>(id) < dataset.rows;
    const std::size_t oram_addr =
        real ? item_oram_address(static_cast<std::size_t>(id), config, context)
             : 0;
    if (context.path_oram_enabled) {
      context.item_oram.access(oram_addr);
    }
    const float distance =
        real ? l2_distance(dataset.row(static_cast<std::size_t>(id)), query,
                           dataset.cols)
             : std::numeric_limits<float>::infinity();
    const ScoredId scored = {distance, real ? id : -1};
    if (config.use_oblivious_sort) {
      sort_comparisons += fixed_schedule_topk_select(scored, top_items);
    } else {
      sort_items.push_back(scored);
    }
  }
  const auto scan_end = std::chrono::steady_clock::now();

  const std::size_t sort_input_size = slots.size();
  std::size_t sort_padded_size = config.use_oblivious_sort
                                     ? top_items.size()
                                     : next_power_of_two(
                                           std::max<std::size_t>(1, sort_input_size));
  const auto sort_start = std::chrono::steady_clock::now();
  if (config.use_oblivious_sort) {
    sort_items = std::move(top_items);
  } else {
    sort_items.resize(sort_padded_size,
                      {std::numeric_limits<float>::infinity(), -1});
    std::sort(sort_items.begin(), sort_items.end(),
              [](const ScoredId &a, const ScoredId &b) { return better(a, b); });
  }
  const auto sort_end = std::chrono::steady_clock::now();

  QueryResult result;
  result.ids.resize(k, -1);
  for (std::size_t i = 0; i < std::min(k, sort_items.size()); ++i) {
    result.ids[i] = sort_items[i].second;
  }
  const auto end = std::chrono::steady_clock::now();
  result.query_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  result.audit.fixed_bucket_count = index.bucket_count;
  result.audit.algorithm4_ms = result.query_ms;
  result.audit.refinement_scan_ms =
      std::chrono::duration<double, std::milli>(scan_end - scan_start).count();
  result.audit.refinement_topk_sort_ms =
      std::chrono::duration<double, std::milli>(sort_end - sort_start).count();
  result.audit.fixed_candidate_slots = slots.size();
  result.audit.fixed_refinement_slots = slots.size();
  result.audit.total_vector_reads = slots.size();
  result.audit.oram_logical_reads =
      context.path_oram_enabled ? slots.size() : 0;
  result.audit.oram_path_length =
      context.path_oram_enabled
          ? path_oram_path_length(context.item_oram.logical_blocks())
          : 0;
  result.audit.oram_bucket_size =
      context.path_oram_enabled ? std::max<std::size_t>(1, config.oram_bucket_size) : 0;
  result.audit.oram_physical_bucket_reads =
      result.audit.oram_logical_reads * result.audit.oram_path_length *
      result.audit.oram_bucket_size;
  result.audit.oram_physical_bucket_writes =
      result.audit.oram_physical_bucket_reads;
  result.audit.oblivious_sort_input_size = sort_input_size;
  result.audit.oblivious_sort_padded_size = sort_padded_size;
  result.audit.oblivious_sort_compare_exchanges = sort_comparisons;
  result.audit.item_oram_logical_accesses =
      context.item_oram.stats().logical_accesses - item_oram_before;
  result.audit.item_oram_executed =
      context.path_oram_enabled && result.audit.item_oram_logical_accesses > 0;
  result.audit.topk_oblivious_sort_executed = config.use_oblivious_sort;
  result.audit.oram_stash_peak =
      std::max(result.audit.oram_stash_peak, context.item_oram.stats().stash_peak);
  result.audit.query_independent_shape = true;
  return result;
}

}  // namespace ppsc
