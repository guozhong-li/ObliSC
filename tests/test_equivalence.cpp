#include "oblisc/oblisc.hpp"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <omp.h>
namespace ppsc {
std::vector<std::int64_t> baseline3(const PublicPaddedIndex &,const float *,const BenchmarkConfig &,PrivateExecutorContext &,AccessAudit &);
QueryResult baseline4(const Matrix &,const float *,const std::vector<std::int64_t> &,const PublicPaddedIndex &,const BenchmarkConfig &,PrivateExecutorContext &);
}
using namespace ppsc;
PrivateExecutorContext context(const BenchmarkConfig &c) {
  PrivateExecutorContext x;x.path_oram_enabled=c.use_path_oram;
  if(c.use_path_oram) {
    x.bucket_oram.reset(c.subspace_num*9,4,42);x.score_oram.reset(16,4,42);x.item_oram.reset(16,4,42);
  }
  return x;
}
int main() try {
  setenv("PPSC_ORAM_LIGHTWEIGHT_THRESHOLD","0",1);omp_set_dynamic(0);
  omp_set_max_active_levels(3);
  for (std::size_t count: {0U,1U,4096U,100000U}) {
    PathOram repeated(16,4,42),batched(16,4,42);
    for (std::size_t i=0;i<count;++i) repeated.access(i%17);
    batched.record_lightweight_accesses(count);
    const auto &a=repeated.stats(); const auto &b=batched.stats();
    if (a.logical_accesses!=b.logical_accesses || a.path_bucket_reads!=b.path_bucket_reads ||
        a.path_bucket_writes!=b.path_bucket_writes || a.stash_peak!=b.stash_peak || a.executed!=b.executed)
      throw std::runtime_error("bulk bookkeeping mismatch");
  }
  setenv("PPSC_ORAM_LIGHTWEIGHT_THRESHOLD","100",1);
  { PathOram protected_backend(16,4,42); bool rejected=false;
    try { protected_backend.record_lightweight_accesses(1); }
    catch (const std::runtime_error &) {rejected=true;}
    if (!rejected) throw std::runtime_error("bulk operation accepted protected backend");
  }
  setenv("PPSC_ORAM_LIGHTWEIGHT_THRESHOLD","0",1);
  std::size_t tests=0;
  for(std::size_t n:{39U,257U}) for(std::size_t ns:{2U,10U}) for(std::size_t dim:{8*ns,10*ns})
  for(bool counter:{false,true}) for(unsigned seed=0;seed<7;++seed) {
    BenchmarkConfig c;c.dataset_size=n;c.dim=dim;c.subspace_num=ns;c.subspace_dimensionality=dim/ns;
    c.kmeans_num_centroid=3;c.k=5;c.alpha=.05;c.beta=.02;c.sc_budget_policy="fixed";
    c.sc_fixed_cells=seed+1;c.score_selection_policy="heap";c.use_path_oram=counter;
    c.use_oblivious_sort=true;c.use_srht=true;
    PublicPaddedIndex x;x.dataset_size=n;x.subspace_num=ns;x.subspace_dimensionality=dim/ns;
    x.kmeans_dim=dim/(2*ns);x.kmeans_num_centroid=3;x.cell_count=9;
    x.centroids.resize(ns*2*3*x.kmeans_dim);
    for(std::size_t i=0;i<x.centroids.size();++i)x.centroids[i]=seed?float((i*7+seed)%11)/11:0;
    x.bucket_offsets.push_back(0);
    for(std::size_t s=0;s<ns;++s)for(std::size_t cell=0;cell<9;++cell) {
      unsigned count=0;
      for(std::size_t id=0;id<n;++id)if((id*(s+1)+seed)%7==cell){x.padded_bucket_ids.push_back(id);++count;}
      x.bucket_counts.push_back(count);
      for(;count<40;++count)x.padded_bucket_ids.push_back(-1);
      x.bucket_offsets.push_back(x.padded_bucket_ids.size());
    }
    Matrix b;b.rows=n;b.cols=dim;b.values.resize(n*dim);
    for(std::size_t i=0;i<b.values.size();++i)b.values[i]=seed?float((i*3+seed)%13)/13:0;
    auto ctx=context(c);AccessAudit a;
    auto slots=baseline3(x,b.row(seed),c,ctx,a);
    auto result=baseline4(b,b.row(seed),slots,x,c,ctx);
    for(int p:{1,2,3,4,5}) {
      c.workers_per_subspace=p;
      c.intra_query_threads=int(x.subspace_num)*p;auto pc=context(c);AccessAudit pa;
      auto ps=algorithm3_fo_candidate_activation(x,b.row(seed),c,pc,pa);
      auto pr=algorithm4_fo_topk_refinement(b,b.row(seed),ps,x,c,pc);
      if(ps!=slots || pr.ids!=result.ids || pa.fixed_candidate_slots!=a.fixed_candidate_slots ||
         pa.bucket_oram_logical_accesses!=a.bucket_oram_logical_accesses ||
         pa.score_oram_logical_accesses!=a.score_oram_logical_accesses ||
         pr.audit.item_oram_logical_accesses!=result.audit.item_oram_logical_accesses)
        throw std::runtime_error("serial/parallel values or legacy counters differ");
      ++tests;
    }
  }
  std::cout<<"{\"tests\":"<<tests<<",\"direct_and_legacy_counter\":true,\"passed\":true}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
