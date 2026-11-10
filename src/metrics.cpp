#include "ppsc/ppsc.hpp"

#include <algorithm>
#include <unordered_set>

namespace ppsc {

double recall_at_k(const GroundTruth &truth, const std::vector<QueryResult> &results,
                   std::size_t k) {
  if (truth.rows == 0 || results.empty() || k == 0) {
    return 0.0;
  }
  const std::size_t q = std::min(truth.rows, results.size());
  std::size_t hits = 0;
  for (std::size_t i = 0; i < q; ++i) {
    std::unordered_set<std::int64_t> expected;
    expected.reserve(k * 2);
    const auto *truth_row = truth.row(i);
    for (std::size_t j = 0; j < std::min(k, truth.k); ++j) {
      expected.insert(truth_row[j]);
    }
    for (std::size_t j = 0; j < std::min(k, results[i].ids.size()); ++j) {
      if (expected.find(results[i].ids[j]) != expected.end()) {
        ++hits;
      }
    }
  }
  return static_cast<double>(hits) / static_cast<double>(q * k);
}

}  // namespace ppsc
