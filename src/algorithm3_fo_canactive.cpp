#include "oblisc/oblisc.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ppsc {

namespace {

struct CellDistance {
  float distance = 0.0F;
  std::size_t cell = 0;
};

struct ScoreEntry {
  std::uint32_t score = 0;
  std::uint32_t best_rank = std::numeric_limits<std::uint32_t>::max();
  std::int64_t id = -1;
};

bool is_power_of_two(std::size_t n) {
  return n > 0 && (n & (n - 1U)) == 0;
}

std::size_t floor_log2(std::size_t n) {
  std::size_t r = 0;
  while (n > 1) {
    ++r;
    n >>= 1U;
  }
  return r;
}

void fwht_inplace(float *vec, std::size_t n) {
  for (std::size_t len = 1; len < n; len <<= 1U) {
    for (std::size_t i = 0; i < n; i += len << 1U) {
      for (std::size_t j = 0; j < len; ++j) {
        const float u = vec[i + j];
        const float v = vec[i + j + len];
        vec[i + j] = u + v;
        vec[i + j + len] = u - v;
      }
    }
  }
}

void kacs_walk(float *data, std::size_t len) {
  const std::size_t half = len / 2;
  for (std::size_t i = 0; i < half; ++i) {
    const float a = data[i];
    const float b = data[i + half];
    data[i] = a + b;
    data[i + half] = a - b;
  }
}

std::vector<float> srht_query(const float *query, std::size_t dim) {
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> coin(0, 1);

  if (is_power_of_two(dim)) {
    std::vector<float> out(dim);
    for (std::size_t i = 0; i < dim; ++i) {
      out[i] = query[i] * (coin(rng) ? 1.0F : -1.0F);
    }
    fwht_inplace(out.data(), dim);
    const float scale = 1.0F / std::sqrt(static_cast<float>(dim));
    for (float &value : out) {
      value *= scale;
    }
    return out;
  }

  const std::size_t trunc = std::size_t{1} << floor_log2(dim);
  const std::size_t padded = dim % 2 == 0 ? dim : dim + 1;
  const std::size_t start = padded - trunc;
  const float scale = 1.0F / std::sqrt(static_cast<float>(trunc));

  std::vector<std::vector<float>> signs(4, std::vector<float>(padded));
  for (auto &round : signs) {
    for (float &sign : round) {
      sign = coin(rng) ? 1.0F : -1.0F;
    }
  }

  std::vector<float> buf(padded, 0.0F);
  std::copy(query, query + dim, buf.begin());

  for (std::size_t round = 0; round < 4; ++round) {
    for (std::size_t j = 0; j < padded; ++j) {
      buf[j] *= signs[round][j];
    }
    float *base = (round % 2 == 0) ? buf.data() : buf.data() + start;
    fwht_inplace(base, trunc);
    const std::size_t begin = (round % 2 == 0) ? 0 : start;
    for (std::size_t j = begin; j < begin + trunc; ++j) {
      buf[j] *= scale;
    }
    kacs_walk(buf.data(), padded);
  }

  std::vector<float> out(dim);
  for (std::size_t i = 0; i < dim; ++i) {
    out[i] = buf[i] * 0.25F;
  }
  return out;
}

float l2_distance(const float *a, const float *b, std::size_t dim) {
  float sum = 0.0F;
  for (std::size_t i = 0; i < dim; ++i) {
    const float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

std::size_t ceil_to_size(double value) {
  if (value <= 0.0) {
    return 0;
  }
  return static_cast<std::size_t>(std::ceil(value));
}

std::size_t next_power_of_two(std::size_t value) {
  std::size_t out = 1;
  while (out < std::max<std::size_t>(1, value)) {
    out <<= 1U;
  }
  return out;
}

bool better_cell(const CellDistance &a, const CellDistance &b) {
  if (a.distance != b.distance) {
    return a.distance < b.distance;
  }
  return a.cell < b.cell;
}

bool better_score(const ScoreEntry &a, const ScoreEntry &b) {
  if (a.score != b.score) {
    return a.score > b.score;
  }
  if (a.best_rank != b.best_rank) {
    return a.best_rank < b.best_rank;
  }
  return a.id < b.id;
}

std::size_t heap_topr_select(const std::vector<std::uint8_t> &scores,
                             const std::vector<std::uint32_t> &best_rank,
                             std::size_t refinement_slots,
                             std::vector<ScoreEntry> &out) {
  out.clear();
  out.reserve(refinement_slots);
  std::size_t comparisons = 0;
  for (std::size_t id = 0; id < scores.size(); ++id) {
    if (scores[id] == 0) {
      continue;
    }
    ScoreEntry entry{scores[id], best_rank[id], static_cast<std::int64_t>(id)};
    if (out.size() < refinement_slots) {
      out.push_back(entry);
      std::push_heap(out.begin(), out.end(), better_score);
      ++comparisons;
      continue;
    }
    ++comparisons;
    if (!out.empty() && better_score(entry, out.front())) {
      std::pop_heap(out.begin(), out.end(), better_score);
      out.back() = entry;
      std::push_heap(out.begin(), out.end(), better_score);
      comparisons +=
          2 * static_cast<std::size_t>(
                  std::ceil(std::log2(static_cast<double>(
                      std::max<std::size_t>(2, out.size())))));
    }
  }
  std::sort(out.begin(), out.end(), better_score);
  return comparisons;
}

std::size_t score_oram_address(std::size_t id, const BenchmarkConfig &config,
                               const PrivateExecutorContext &context) {
  if (!context.path_oram_enabled || context.score_oram.logical_blocks() == 0) {
    return id;
  }
  if (config.score_selection_policy == "heap" ||
      config.score_selection_policy == "topr_heap" ||
      config.score_selection_policy == "topr-heap" ||
      config.score_selection_policy == "selection") {
    const std::uint64_t mixed =
        (static_cast<std::uint64_t>(id) + 0x9e3779b97f4a7c15ULL) *
        0xbf58476d1ce4e5b9ULL;
    return static_cast<std::size_t>(
        mixed % static_cast<std::uint64_t>(context.score_oram.logical_blocks()));
  }
  return id;
}

template <typename T, typename Better>
std::size_t bitonic_oblivious_sort(std::vector<T> &items, Better better) {
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

std::size_t clamp_iterations(std::size_t value, std::size_t cell_count) {
  return std::min(cell_count, std::max<std::size_t>(1, value));
}

std::size_t positive_bucket_quantile(const PublicPaddedIndex &index,
                                     std::size_t subspace, double quantile) {
  const std::size_t begin = subspace * index.cell_count;
  const std::size_t end = begin + index.cell_count;
  std::vector<std::uint32_t> counts;
  counts.reserve(index.cell_count);
  for (std::size_t bucket = begin; bucket < end; ++bucket) {
    const std::uint32_t count = index.bucket_counts[bucket];
    if (count > 0) {
      counts.push_back(count);
    }
  }
  if (counts.empty()) {
    return 1;
  }
  std::sort(counts.begin(), counts.end());
  const double q = std::min(1.0, std::max(0.0, quantile));
  const std::size_t pos =
      q <= 0.0
          ? 0
          : std::min(counts.size() - 1,
                     static_cast<std::size_t>(
                         std::ceil(q * static_cast<double>(counts.size()))) -
                         1);
  return std::max<std::size_t>(1, counts[pos]);
}

std::size_t activation_iterations(const PublicPaddedIndex &index,
                                  const BenchmarkConfig &config,
                                  std::size_t subspace,
                                  std::size_t collision_target) {
  const std::size_t raw_min_cell_size =
      subspace < index.subspace_min_cell_sizes.size()
          ? std::max<std::size_t>(1, index.subspace_min_cell_sizes[subspace])
          : 1;

  if (config.sc_budget_policy == "strict_min" ||
      config.sc_budget_policy == "strict-min" ||
      config.sc_budget_policy == "min") {
    return clamp_iterations(
        ceil_to_size(static_cast<double>(collision_target) /
                     static_cast<double>(raw_min_cell_size)),
        index.cell_count);
  }

  if (config.sc_budget_policy == "clipped" ||
      config.sc_budget_policy == "clipped_floor" ||
      config.sc_budget_policy == "clipped-floor") {
    const double ratio = config.sc_floor_ratio > 0.0 ? config.sc_floor_ratio : 0.5;
    const double average_bucket_size =
        static_cast<double>(index.dataset_size) /
        static_cast<double>(std::max<std::size_t>(1, index.cell_count));
    const std::size_t public_floor =
        std::max<std::size_t>(1, ceil_to_size(ratio * average_bucket_size));
    const std::size_t denominator =
        std::max(raw_min_cell_size, public_floor);
    return clamp_iterations(
        ceil_to_size(static_cast<double>(collision_target) /
                     static_cast<double>(denominator)),
        index.cell_count);
  }

  if (config.sc_budget_policy == "quantile" ||
      config.sc_budget_policy == "quantile_floor" ||
      config.sc_budget_policy == "quantile-floor") {
    const std::size_t public_floor =
        positive_bucket_quantile(index, subspace, config.sc_quantile);
    const std::size_t denominator =
        std::max(raw_min_cell_size, public_floor);
    return clamp_iterations(
        ceil_to_size(static_cast<double>(collision_target) /
                     static_cast<double>(denominator)),
        index.cell_count);
  }

  if (config.sc_budget_policy == "fixed" ||
      config.sc_budget_policy == "fixed_budget" ||
      config.sc_budget_policy == "fixed-budget" ||
      config.sc_budget_policy == "calibrated_fixed" ||
      config.sc_budget_policy == "calibrated-fixed") {
    const std::size_t fixed_cells =
        config.sc_fixed_cells == 0
            ? ceil_to_size(config.alpha * static_cast<double>(index.cell_count))
            : config.sc_fixed_cells;
    return clamp_iterations(fixed_cells, index.cell_count);
  }

  throw std::runtime_error("unknown SC budget policy: " +
                           config.sc_budget_policy);
}

}  // namespace

std::vector<std::int64_t> algorithm3_fo_candidate_activation(
    const PublicPaddedIndex &index, const float *query,
    const BenchmarkConfig &config, PrivateExecutorContext &context,
    AccessAudit &audit) {
  const auto algorithm_start = std::chrono::steady_clock::now();
  const auto projection_start = std::chrono::steady_clock::now();
  const std::vector<float> projected =
      config.use_srht ? srht_query(query, config.dim)
                      : std::vector<float>(query, query + config.dim);
  const auto projection_end = std::chrono::steady_clock::now();
  audit.candidate_projection_ms =
      std::chrono::duration<double, std::milli>(projection_end -
                                                projection_start)
          .count();

  const std::size_t collision_target =
      std::max<std::size_t>(1, ceil_to_size(config.alpha * index.dataset_size));
  const std::size_t refinement_slots =
      std::max<std::size_t>(config.k,
                            ceil_to_size(config.beta * index.dataset_size));

  std::vector<std::uint8_t> scores(index.dataset_size, 0);
  std::vector<std::uint32_t> best_rank(
      index.dataset_size, std::numeric_limits<std::uint32_t>::max());
  std::size_t fixed_cell_reads = 0;
  std::size_t fixed_candidate_slots = 0;
  std::size_t bucket_oram_before = context.bucket_oram.stats().logical_accesses;
  std::size_t score_oram_before = context.score_oram.stats().logical_accesses;

  std::vector<CellDistance> cells;
  cells.reserve(next_power_of_two(index.cell_count));

  for (std::size_t subspace = 0; subspace < index.subspace_num; ++subspace) {
    const std::size_t iterations =
        activation_iterations(index, config, subspace, collision_target);
    fixed_cell_reads += iterations;

    const float *query_first =
        projected.data() + subspace * index.subspace_dimensionality;
    const float *query_second = query_first + index.kmeans_dim;
    const float *centroids_first =
        index.centroids.data() +
        (subspace * 2) * index.kmeans_num_centroid * index.kmeans_dim;
    const float *centroids_second =
        index.centroids.data() +
        (subspace * 2 + 1) * index.kmeans_num_centroid * index.kmeans_dim;

    std::vector<float> first_dist(index.kmeans_num_centroid);
    std::vector<float> second_dist(index.kmeans_num_centroid);
    for (std::size_t c = 0; c < index.kmeans_num_centroid; ++c) {
      first_dist[c] = l2_distance(query_first,
                                  centroids_first + c * index.kmeans_dim,
                                  index.kmeans_dim);
      second_dist[c] = l2_distance(query_second,
                                   centroids_second + c * index.kmeans_dim,
                                   index.kmeans_dim);
    }

    cells.clear();
    for (std::size_t u = 0; u < index.kmeans_num_centroid; ++u) {
      for (std::size_t v = 0; v < index.kmeans_num_centroid; ++v) {
        cells.push_back({first_dist[u] + second_dist[v],
                         u * index.kmeans_num_centroid + v});
      }
    }
    const std::size_t cell_input_size = cells.size();
    const std::size_t cell_padded_size = next_power_of_two(cell_input_size);
    cells.resize(cell_padded_size,
                 {std::numeric_limits<float>::infinity(),
                  std::numeric_limits<std::size_t>::max()});
    audit.cell_sort_input_size += cell_input_size;
    audit.cell_sort_padded_size += cell_padded_size;
    const auto cell_sort_start = std::chrono::steady_clock::now();
    audit.cell_sort_compare_exchanges +=
        bitonic_oblivious_sort(cells, better_cell);
    const auto cell_sort_end = std::chrono::steady_clock::now();
    audit.candidate_cell_sort_ms +=
        std::chrono::duration<double, std::milli>(cell_sort_end -
                                                  cell_sort_start)
            .count();
    audit.cell_oblivious_sort_executed = true;

    const auto bucket_scan_start = std::chrono::steady_clock::now();
    for (std::size_t pos = 0; pos < iterations; ++pos) {
      const std::size_t bucket = subspace * index.cell_count + cells[pos].cell;
      if (context.path_oram_enabled) {
        context.bucket_oram.access(bucket);
      }
      const std::uint64_t begin = index.bucket_offsets[bucket];
      const std::uint64_t end = index.bucket_offsets[bucket + 1];
      fixed_candidate_slots += static_cast<std::size_t>(end - begin);
      const std::uint32_t activation_rank =
          static_cast<std::uint32_t>(subspace * index.cell_count + pos);
      for (std::uint64_t offset = begin; offset < end; ++offset) {
        const std::int32_t id =
            index.padded_bucket_ids[static_cast<std::size_t>(offset)];
        const std::size_t oram_addr =
            id >= 0 && static_cast<std::size_t>(id) < scores.size()
                ? score_oram_address(static_cast<std::size_t>(id), config,
                                     context)
                : 0;
        if (context.path_oram_enabled) {
          context.score_oram.access(oram_addr);
        }
        if (id >= 0 && static_cast<std::size_t>(id) < scores.size()) {
          const std::size_t uid = static_cast<std::size_t>(id);
          if (scores[uid] < std::numeric_limits<std::uint8_t>::max()) {
            ++scores[uid];
          }
          best_rank[uid] = std::min(best_rank[uid], activation_rank);
        }
      }
    }
    const auto bucket_scan_end = std::chrono::steady_clock::now();
    audit.candidate_bucket_scan_ms +=
        std::chrono::duration<double, std::milli>(bucket_scan_end -
                                                  bucket_scan_start)
            .count();
  }

  std::vector<std::int64_t> slots(refinement_slots, -1);

  std::vector<ScoreEntry> score_entries;
  const std::size_t score_input_size = index.dataset_size;
  audit.score_sort_input_size = score_input_size;
  const auto score_sort_start = std::chrono::steady_clock::now();
  if (config.score_selection_policy == "heap" ||
      config.score_selection_policy == "topr_heap" ||
      config.score_selection_policy == "topr-heap" ||
      config.score_selection_policy == "selection") {
    audit.score_sort_padded_size = refinement_slots;
    audit.score_sort_compare_exchanges =
        heap_topr_select(scores, best_rank, refinement_slots, score_entries);
  } else {
    const std::size_t score_padded_size = next_power_of_two(score_input_size);
    score_entries.reserve(score_padded_size);
    for (std::size_t id = 0; id < index.dataset_size; ++id) {
      score_entries.push_back({scores[id], best_rank[id],
                               static_cast<std::int64_t>(id)});
    }
    score_entries.resize(
        score_padded_size,
        {0, std::numeric_limits<std::uint32_t>::max(),
         std::numeric_limits<std::int64_t>::max()});
    audit.score_sort_padded_size = score_padded_size;
    audit.score_sort_compare_exchanges =
        bitonic_oblivious_sort(score_entries, better_score);
  }
  const auto score_sort_end = std::chrono::steady_clock::now();
  audit.candidate_score_sort_ms =
      std::chrono::duration<double, std::milli>(score_sort_end -
                                                score_sort_start)
          .count();
  audit.score_oblivious_sort_executed = true;

  for (std::size_t i = 0; i < slots.size(); ++i) {
    if (i < score_entries.size() && score_entries[i].id >= 0 &&
        score_entries[i].id < static_cast<std::int64_t>(index.dataset_size)) {
      slots[i] = score_entries[i].id;
    }
  }

  audit.fixed_bucket_count = fixed_cell_reads;
  audit.fixed_candidate_slots = fixed_candidate_slots;
  audit.fixed_refinement_slots = refinement_slots;
  audit.bucket_oram_logical_accesses =
      context.bucket_oram.stats().logical_accesses - bucket_oram_before;
  audit.score_oram_logical_accesses =
      context.score_oram.stats().logical_accesses - score_oram_before;
  audit.bucket_oram_executed = context.path_oram_enabled &&
                               audit.bucket_oram_logical_accesses > 0;
  audit.score_oram_executed = context.path_oram_enabled &&
                              audit.score_oram_logical_accesses > 0;
  audit.oram_stash_peak =
      std::max(context.bucket_oram.stats().stash_peak,
               context.score_oram.stats().stash_peak);
  audit.query_independent_shape = true;
  const auto algorithm_end = std::chrono::steady_clock::now();
  audit.algorithm3_ms =
      std::chrono::duration<double, std::milli>(algorithm_end -
                                                algorithm_start)
          .count();
  return slots;
}

}  // namespace ppsc
