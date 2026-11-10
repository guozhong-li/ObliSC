#include "ppsc/ppsc.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>
#include <omp.h>
#ifdef REUSE_WORKSPACE
#define QUERY_STORAGE thread_local
#else
#define QUERY_STORAGE
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif

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

std::size_t parallel_topr(const std::vector<std::uint8_t> &scores,
                         const std::vector<std::uint32_t> &ranks,
                         std::size_t limit, int workers,
                         std::vector<ScoreEntry> &out, int &observed) {
  std::vector<std::vector<ScoreEntry>> pieces(workers);
  std::size_t considered = 0;
#pragma omp parallel for schedule(static) num_threads(workers) reduction(+:considered)
  for (int part = 0; part < workers; ++part) {
    if (part == 0) observed = omp_get_num_threads();
    auto &v = pieces[part];
    v.reserve(std::min(limit, scores.size()/workers+1));
    const std::size_t begin = scores.size()*part/workers;
    const std::size_t end = scores.size()*(part+1)/workers;
    for (std::size_t id = begin; id < end; ++id) {
      if (!scores[id]) continue;
      ScoreEntry e{scores[id], ranks[id], static_cast<std::int64_t>(id)};
      ++considered;
      if (v.size() < limit) {
        v.push_back(e); std::push_heap(v.begin(), v.end(), better_score);
      } else if (better_score(e, v.front())) {
        std::pop_heap(v.begin(), v.end(), better_score);
        v.back() = e; std::push_heap(v.begin(), v.end(), better_score);
      }
    }
    std::sort(v.begin(), v.end(), better_score);
  }
  // A global top-R item must occur in its own partition's top R.
  using Cursor = std::pair<std::size_t,std::size_t>;
  auto worse = [&](const Cursor &a, const Cursor &b) {
    return better_score(pieces[b.first][b.second], pieces[a.first][a.second]);
  };
  std::priority_queue<Cursor,std::vector<Cursor>,decltype(worse)> heap(worse);
  for (std::size_t p = 0; p < pieces.size(); ++p)
    if (!pieces[p].empty()) heap.emplace(p,0);
  out.clear(); out.reserve(limit);
  while (!heap.empty() && out.size() < limit) {
    auto c = heap.top(); heap.pop(); out.push_back(pieces[c.first][c.second]);
    if (++c.second < pieces[c.first].size()) heap.push(c);
  }
  return considered;
}

unsigned packed_score(std::uint64_t value) {
  return std::min(255U, static_cast<unsigned>(value));
}

void exact_topr(const std::vector<std::uint64_t> &state, std::size_t rank_count,
                std::size_t limit, int workers, std::vector<ScoreEntry> &out,
                int &observed) {
  QUERY_STORAGE std::vector<std::array<std::size_t,256>> counts_storage;
  auto &counts = counts_storage;
  counts.resize(workers);
#pragma omp parallel for schedule(static) num_threads(workers)
  for (int part=0; part<workers; ++part) {
    if (part==0) observed=omp_get_num_threads();
    counts[part].fill(0);
    for (std::size_t id=state.size()*part/workers; id<state.size()*(part+1)/workers; ++id)
      ++counts[part][packed_score(state[id])];
  }
  std::size_t remaining=limit;
  unsigned threshold=1;
  for (unsigned score=255; score>0; --score) {
    std::size_t count=0;
    for (int p=0;p<workers;++p) count+=counts[p][score];
    if (count>=remaining || score==1) {threshold=score;break;}
    remaining-=count;
  }
  QUERY_STORAGE std::vector<std::vector<std::size_t>> ranks_storage;
  auto &ranks = ranks_storage;
  ranks.resize(workers);
  for (auto &row : ranks) row.assign(rank_count, 0);
#pragma omp parallel for schedule(static) num_threads(workers)
  for (int part=0;part<workers;++part) {
    for (std::size_t id=state.size()*part/workers;id<state.size()*(part+1)/workers;++id) {
      const auto value=state[id];
      if (packed_score(value)==threshold) ++ranks[part][value>>32];
    }
  }
  std::size_t boundary=rank_count;
  for (std::size_t rank=0;rank<rank_count;++rank) {
    std::size_t count=0;
    for (int p=0;p<workers;++p) count+=ranks[p][rank];
    if (count>=remaining) {boundary=rank;break;}
    remaining-=count;
  }
  // Contiguous ID partitions preserve the final ID tie break at the boundary.
  QUERY_STORAGE std::vector<std::vector<ScoreEntry>> better_storage, equal_storage;
  auto &better = better_storage;
  auto &equal = equal_storage;
  better.resize(workers); equal.resize(workers);
  for (auto &row : better) row.clear();
  for (auto &row : equal) row.clear();
#pragma omp parallel for schedule(static) num_threads(workers)
  for (int part=0;part<workers;++part) {
    for (std::size_t id=state.size()*part/workers;id<state.size()*(part+1)/workers;++id) {
      const auto value=state[id]; const auto score=packed_score(value);
      const auto rank=static_cast<std::uint32_t>(value>>32);
      if (!score) continue;
      const ScoreEntry entry{score,rank,static_cast<std::int64_t>(id)};
      if (score>threshold || (score==threshold && rank<boundary)) better[part].push_back(entry);
      else if (score==threshold && rank==boundary && equal[part].size()<remaining)
        equal[part].push_back(entry);
    }
  }
  out.clear();out.reserve(limit);
  for (auto &v:better) out.insert(out.end(),v.begin(),v.end());
  if (out.size()>limit) throw std::runtime_error("invalid exact selection threshold");
  for (auto &v:equal) {
    const auto count=std::min(v.size(),limit-out.size());
    out.insert(out.end(),v.begin(),v.begin()+count);
  }
  std::sort(out.begin(),out.end(),better_score);
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

std::size_t grouped_cell_sort(std::vector<CellDistance> &cells, int workers) {
  if (workers == 1) return bitonic_oblivious_sort(cells, better_cell);
  const auto n = cells.size();
  int observed = 0;
#pragma omp parallel num_threads(workers)
  {
#pragma omp single
    observed = omp_get_num_threads();
    for (std::size_t size = 2; size <= n; size <<= 1U) {
      for (std::size_t stride = size >> 1U; stride > 0; stride >>= 1U) {
#pragma omp for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
          const auto j = i ^ stride;
          if (j <= i) continue;
          const bool ascending = (i & size) == 0;
          if (ascending ? better_cell(cells[j], cells[i]) : better_cell(cells[i], cells[j]))
            std::swap(cells[i], cells[j]);
        }
      }
    }
  }
  if (observed != workers) throw std::runtime_error("subspace sort team reduced");
  const auto levels = floor_log2(n);
  return (n / 2) * levels * (levels + 1) / 2;
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

  const auto init_start = std::chrono::steady_clock::now();
  if (index.subspace_num > std::numeric_limits<std::uint32_t>::max()/index.cell_count)
    throw std::runtime_error("activation rank exceeds checked width");
  // High word is the minimum activation rank, low word the exact vote count.
  // One CAS updates both, irrespective of the order in which cells finish.
  // Capture a normal reference: OpenMP workers must use the caller's storage,
  // not resolve the thread-local name to a different worker's empty vector.
  QUERY_STORAGE std::vector<std::uint64_t> state_storage;
  auto &state = state_storage;
  state.assign(index.dataset_size, UINT64_C(0xffffffff00000000));
  std::size_t fixed_cell_reads = 0;
  std::size_t fixed_candidate_slots = 0;
  std::size_t bucket_oram_before = context.bucket_oram.stats().logical_accesses;
  std::size_t score_oram_before = context.score_oram.stats().logical_accesses;

  const int workers = config.intra_query_threads;
  const int group_workers = config.workers_per_subspace;
  if (group_workers < 1 || group_workers > 5 || workers != int(index.subspace_num) * group_workers)
    throw std::runtime_error("invalid public subspace group size");
  if (workers < 1 || workers > 100 || (context.path_oram_enabled &&
      (!context.bucket_oram.lightweight() || !context.score_oram.lightweight()))) {
    throw std::runtime_error("only direct/legacy-counter backend supported");
  }
  QUERY_STORAGE std::vector<PrivateExecutorContext> scan_contexts_storage;
  auto &scan_contexts = scan_contexts_storage;
  scan_contexts.clear(); scan_contexts.resize(workers);
  for (auto &local : scan_contexts) {
    local.path_oram_enabled = context.path_oram_enabled;
    if (local.path_oram_enabled)
      local.score_oram.reset(context.score_oram.logical_blocks(),config.oram_bucket_size,42);
  }
  QUERY_STORAGE std::vector<AccessAudit> partial_audit_storage;
  QUERY_STORAGE std::vector<std::size_t> partial_reads_storage, partial_slots_storage;
  QUERY_STORAGE std::vector<std::vector<CellDistance>> ordered_cells_storage;
  auto &partial_audit = partial_audit_storage;
  auto &partial_reads = partial_reads_storage;
  auto &partial_slots = partial_slots_storage;
  auto &ordered_cells = ordered_cells_storage;
  partial_audit.assign(index.subspace_num, AccessAudit{});
  partial_reads.assign(index.subspace_num, 0);
  partial_slots.assign(index.subspace_num, 0);
  ordered_cells.resize(index.subspace_num);
  const auto cells_start = std::chrono::steady_clock::now();
  audit.state_init_ms = std::chrono::duration<double,std::milli>(cells_start-init_start).count();
#pragma omp parallel for schedule(static) num_threads(index.subspace_num)
  for (std::size_t subspace = 0; subspace < index.subspace_num; ++subspace) {
    auto &audit = partial_audit[subspace];
    auto &fixed_cell_reads = partial_reads[subspace];
    auto &cells = ordered_cells[subspace];
    cells.reserve(next_power_of_two(index.cell_count));
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

    QUERY_STORAGE std::vector<float> first_dist_storage, second_dist_storage;
    auto &first_dist = first_dist_storage;
    auto &second_dist = second_dist_storage;
    first_dist.resize(index.kmeans_num_centroid);
    second_dist.resize(index.kmeans_num_centroid);
    int distance_workers = 0;
#pragma omp parallel num_threads(group_workers) if(group_workers > 1)
    {
#pragma omp single
      distance_workers = omp_get_num_threads();
#pragma omp for schedule(static)
    for (std::size_t c = 0; c < index.kmeans_num_centroid; ++c) {
      first_dist[c] = l2_distance(query_first,
                                  centroids_first + c * index.kmeans_dim,
                                  index.kmeans_dim);
      second_dist[c] = l2_distance(query_second,
                                   centroids_second + c * index.kmeans_dim,
                                   index.kmeans_dim);
    }
    }
    if (distance_workers != group_workers)
      throw std::runtime_error("subspace distance team reduced");

    cells.resize(index.cell_count);
#pragma omp parallel for schedule(static) num_threads(group_workers) if(group_workers > 1)
    for (std::size_t u = 0; u < index.kmeans_num_centroid; ++u) {
      for (std::size_t v = 0; v < index.kmeans_num_centroid; ++v) {
        cells[u * index.kmeans_num_centroid + v] = {first_dist[u] + second_dist[v],
                         u * index.kmeans_num_centroid + v};
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
        grouped_cell_sort(cells, group_workers);
    const auto cell_sort_end = std::chrono::steady_clock::now();
    audit.candidate_cell_sort_ms +=
        std::chrono::duration<double, std::milli>(cell_sort_end -
                                                  cell_sort_start)
            .count();
    audit.cell_oblivious_sort_executed = true;

  }

  const auto tasks_start = std::chrono::steady_clock::now();
  audit.cell_prepare_ms = std::chrono::duration<double,std::milli>(tasks_start-cells_start).count();
  struct Scan { std::size_t subspace, pos; std::uint64_t begin, end; };
  QUERY_STORAGE std::vector<Scan> tasks_storage;
  auto &tasks = tasks_storage;
  tasks.clear();
  for (std::size_t s = 0; s < index.subspace_num; ++s) {
    for (std::size_t pos = 0; pos < partial_reads[s]; ++pos) {
      const auto b = s*index.cell_count+ordered_cells[s][pos].cell;
      if (context.path_oram_enabled) context.bucket_oram.access(b);
      const auto begin = index.bucket_offsets[b], end = index.bucket_offsets[b+1];
      partial_slots[s] += end-begin;
      for (auto offset = begin; offset < end; offset += 4096)
        tasks.push_back({s,pos,offset,std::min(end,offset+4096)});
    }
  }
  const auto scan_start = std::chrono::steady_clock::now();
  audit.scan_prepare_ms = std::chrono::duration<double,std::milli>(scan_start-tasks_start).count();
  // Algorithm 2 places each ID exactly once per subspace. Disjoint slot
  // ranges therefore own disjoint real-ID destinations within each row.
#pragma omp parallel for schedule(static) num_threads(workers)
  for (std::size_t t = 0; t < tasks.size(); ++t) {
    if (t == 0) audit.scan_workers_observed = omp_get_num_threads();
    const auto task = tasks[t];
    auto &local = scan_contexts[omp_get_thread_num()];
    const auto r = static_cast<std::uint32_t>(task.subspace*index.cell_count+task.pos);
    if (local.path_oram_enabled)
      local.score_oram.record_lightweight_accesses(task.end-task.begin);
    const auto vote = [&](std::int32_t id) {
      if (id >= 0 && static_cast<std::size_t>(id) < index.dataset_size) {
        auto previous=__atomic_load_n(&state[id],__ATOMIC_RELAXED);
        std::uint64_t next;
        do {
          const auto rank=std::min(static_cast<std::uint32_t>(previous>>32),r);
          next=(static_cast<std::uint64_t>(rank)<<32) |
               (static_cast<std::uint32_t>(previous)+1U);
        } while (!__atomic_compare_exchange_n(&state[id],&previous,next,false,
                                               __ATOMIC_RELAXED,__ATOMIC_RELAXED));
      }
    };
    auto offset=task.begin;
#ifdef __AVX2__
    // Direct-memory only: read every slot, bypass scalar work on dummy lanes.
    for (;task.end-offset>=8;offset+=8) {
      const auto ids=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(index.padded_bucket_ids.data()+offset));
      unsigned mask=static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(
          _mm256_cmpgt_epi32(ids,_mm256_set1_epi32(-1)))));
      while (mask) {
        const auto lane=static_cast<unsigned>(__builtin_ctz(mask));
        vote(index.padded_bucket_ids[offset+lane]);
        mask&=mask-1;
      }
    }
#endif
    for (;offset<task.end;++offset) vote(index.padded_bucket_ids[offset]);
  }
  audit.candidate_bucket_scan_ms = std::chrono::duration<double,std::milli>(
      std::chrono::steady_clock::now()-scan_start).count();

  const auto reduce_start = std::chrono::steady_clock::now();
  // The scan's OpenMP barrier completes all atomic writes before selection.
  audit.score_reduce_ms = std::chrono::duration<double,std::milli>(
      std::chrono::steady_clock::now()-reduce_start).count();
  for (std::size_t s = 0; s < index.subspace_num; ++s) {
    fixed_cell_reads += partial_reads[s];
    fixed_candidate_slots += partial_slots[s];
    audit.cell_sort_input_size += partial_audit[s].cell_sort_input_size;
    audit.cell_sort_padded_size += partial_audit[s].cell_sort_padded_size;
    audit.cell_sort_compare_exchanges += partial_audit[s].cell_sort_compare_exchanges;
    audit.candidate_cell_sort_ms += partial_audit[s].candidate_cell_sort_ms;
    audit.candidate_bucket_scan_ms += partial_audit[s].candidate_bucket_scan_ms;
  }
  audit.cell_oblivious_sort_executed = true;
  std::vector<std::int64_t> slots(refinement_slots, -1);

  QUERY_STORAGE std::vector<ScoreEntry> score_entries;
  score_entries.clear();
  const std::size_t score_input_size = index.dataset_size;
  audit.score_sort_input_size = score_input_size;
  const auto score_sort_start = std::chrono::steady_clock::now();
  if (config.score_selection_policy == "heap" ||
      config.score_selection_policy == "topr_heap" ||
      config.score_selection_policy == "topr-heap" ||
      config.score_selection_policy == "selection") {
    audit.score_sort_padded_size = refinement_slots;
#ifdef PACKED_HEAP
    std::vector<std::uint8_t> scores(state.size());
    std::vector<std::uint32_t> ranks(state.size());
#pragma omp parallel for schedule(static) num_threads(workers)
    for (std::size_t id=0;id<state.size();++id) {
      scores[id]=packed_score(state[id]);ranks[id]=state[id]>>32;
    }
    parallel_topr(scores,ranks,refinement_slots,workers,score_entries,audit.selection_workers_observed);
#else
    exact_topr(state,index.subspace_num*index.cell_count,refinement_slots,workers,
               score_entries,audit.selection_workers_observed);
#endif
    // No claim of an oblivious comparison schedule for this direct backend.
    audit.score_sort_compare_exchanges = 0;
  } else {
    const std::size_t score_padded_size = next_power_of_two(score_input_size);
    score_entries.reserve(score_padded_size);
    for (std::size_t id = 0; id < index.dataset_size; ++id) {
      score_entries.push_back({packed_score(state[id]), static_cast<std::uint32_t>(state[id]>>32),
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
  audit.score_oblivious_sort_executed = false;

  for (std::size_t i = 0; i < slots.size(); ++i) {
    if (i < score_entries.size() && score_entries[i].id >= 0 &&
        score_entries[i].id < static_cast<std::int64_t>(index.dataset_size)) {
      slots[i] = score_entries[i].id;
    }
  }

  audit.fixed_bucket_count = fixed_cell_reads;
  audit.fixed_candidate_slots = fixed_candidate_slots;
  audit.fixed_refinement_slots = refinement_slots;
  if (audit.scan_workers_observed != workers || audit.selection_workers_observed != workers)
    throw std::runtime_error("thread team was silently reduced");
  audit.bucket_oram_logical_accesses =
      context.bucket_oram.stats().logical_accesses - bucket_oram_before;
  audit.score_oram_logical_accesses =
      context.score_oram.stats().logical_accesses - score_oram_before;
  for (const auto &local : scan_contexts)
    audit.score_oram_logical_accesses += local.score_oram.stats().logical_accesses;
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
