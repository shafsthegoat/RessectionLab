// Corrected-only ownership transitions with actual SDK types and mocked API calls.
// No Accelerate factorization/solve, FEBio model or original faulty code executes.
#include <Accelerate/Accelerate.h>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <new>
#include <vector>
#include <map>
#include <set>
#include <algorithm>

static void* watched[2] = {};
static bool released[2] = {};
void* operator new(std::size_t n) { if (void* p=std::malloc(n ? n : 1)) return p; throw std::bad_alloc(); }
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept { for(int i=0;i<2;++i) if(p && p==watched[i]) released[i]=true; std::free(p); }
void operator delete[](void* p) noexcept { ::operator delete(p); }
void operator delete(void* p, std::size_t) noexcept { ::operator delete(p); }
void operator delete[](void* p, std::size_t) noexcept { ::operator delete(p); }

namespace ledger {
  enum class Symbolic { success, failed_empty, failed_owned };
  enum class Numeric { success, failed_storage, failed_symbolic_only, failed_empty };
  Symbolic symbolic_mode=Symbolic::success;
  Numeric numeric_mode=Numeric::success;
  std::map<void*,int> symbolic_refs;
  std::set<void*> numeric_storage;
  int symbolic_calls=0,numeric_calls=0,symbolic_cleanups=0,numeric_cleanups=0,solve_calls=0;
  void* allocate() { return new int(17); }
  void release_symbolic(void* pointer) {
    assert(pointer && symbolic_refs.count(pointer));
    assert(symbolic_refs[pointer]>0);
    if(--symbolic_refs[pointer]==0) { symbolic_refs.erase(pointer); delete static_cast<int*>(pointer); }
  }
  void empty() { assert(symbolic_refs.empty()); assert(numeric_storage.empty()); }
  void reset() {
    empty(); symbolic_mode=Symbolic::success; numeric_mode=Numeric::success;
    symbolic_calls=numeric_calls=symbolic_cleanups=numeric_cleanups=solve_calls=0;
  }
}
static SparseOpaqueSymbolicFactorization MockSparseFactor(SparseFactorization_t type,
        SparseMatrixStructure matrix, SparseSymbolicFactorOptions options) {
  using namespace ledger;
  ++symbolic_calls;
  assert(!matrix.attributes.transpose && matrix.attributes.triangle==SparseUpperTriangle);
  assert(matrix.attributes._reserved==0 && !matrix.attributes._allocatedBySparse);
  assert(matrix.blockSize==1 && matrix.columnStarts && matrix.rowIndices);
  assert(matrix.columnStarts[0]==0 && matrix.columnStarts[matrix.columnCount]>=0);
  assert(options.malloc && options.free && options.reportError && !options.order && !options.ignoreRowsAndColumns);
  SparseOpaqueSymbolicFactorization value{};
  value.status=symbolic_mode==Symbolic::success ? SparseStatusOK : SparseParameterError;
  value.rowCount=matrix.rowCount; value.columnCount=matrix.columnCount;
  value.type=type; value.attributes=matrix.attributes; value.blockSize=matrix.blockSize;
  if(symbolic_mode!=Symbolic::failed_empty) { value.factorization=allocate(); symbolic_refs[value.factorization]=1; }
  return value;
}
static SparseOpaqueFactorization_Double MockSparseFactor(SparseOpaqueSymbolicFactorization symbol,
        SparseMatrix_Double matrix) {
  using namespace ledger;
  ++numeric_calls;
  assert(symbol.status==SparseStatusOK && symbolic_refs.count(symbol.factorization));
  assert(numeric_storage.empty()); // previous numerical storage must be released first
  assert(symbolic_refs[symbol.factorization]==1); // previous embedded reference released, too
  assert(matrix.data && matrix.structure.columnStarts);
  SparseOpaqueFactorization_Double value{};
  value.status=numeric_mode==Numeric::success ? SparseStatusOK : SparseFactorizationFailed;
  if(numeric_mode==Numeric::failed_empty) {
    value.symbolicFactorization.status=SparseParameterError; // canonical defined empty failure
    return value;
  }
  value.symbolicFactorization=symbol;
  ++symbolic_refs[symbol.factorization];
  if(numeric_mode!=Numeric::failed_symbolic_only) {
    value.numericFactorization=allocate(); numeric_storage.insert(value.numericFactorization);
  }
  return value;
}
static void MockSparseCleanup(SparseOpaqueSymbolicFactorization value) {
  ++ledger::symbolic_cleanups; ledger::release_symbolic(value.factorization);
}
static void MockSparseCleanup(SparseOpaqueFactorization_Double value) {
  using namespace ledger;
  assert(value.numericFactorization || value.symbolicFactorization.factorization); // no dummy cleanup
  ++numeric_cleanups;
  if(value.numericFactorization) {
    assert(numeric_storage.erase(value.numericFactorization)==1);
    delete static_cast<int*>(value.numericFactorization);
  }
  if(value.symbolicFactorization.factorization) release_symbolic(value.symbolicFactorization.factorization);
}
static void MockSparseSolve(SparseOpaqueFactorization_Double value, DenseVector_Double, DenseVector_Double) {
  assert(value.status==SparseStatusOK && ledger::numeric_storage.count(value.numericFactorization));
  ++ledger::solve_calls; // No numerical solution is computed or asserted.
}
template<class... T> static SparseIterativeStatus_t MockSparseSolve(T...) { std::abort(); }
template<class T> static int ExcludedIterative(T) { std::abort(); }
#define SparseFactor MockSparseFactor
#define SparseCleanup MockSparseCleanup
#define SparseSolve MockSparseSolve
#define SparseConjugateGradient ExcludedIterative
#define SparseGMRES ExcludedIterative
#define SparseLSMR ExcludedIterative

struct FEModel {};
enum Matrix_Type { REAL_SYMMETRIC, REAL_UNSYMMETRIC, REAL_SYMM_STRUCTURE };
struct SparseMatrix { virtual ~SparseMatrix() = default; };
struct CompactMatrix : SparseMatrix {
  int rows=2, columns=2;
  std::vector<int> pointers{0,2,4},indices{0,1,0,1};
  std::vector<double> values{2,1,1,2};
  int Rows() const { return rows; } int Columns() const { return columns; }
  int NonZeroes() const { return static_cast<int>(values.size()); }
  int* Pointers() { return pointers.data(); } int* Indices() { return indices.data(); }
  double* Values() { return values.data(); }
};
struct CCSSparseMatrix : CompactMatrix { explicit CCSSparseMatrix(int) {} };
struct LinearSolver {
  explicit LinearSolver(FEModel*) {} virtual ~LinearSolver()=default;
  bool PreProcess() { return true; } void UpdateStats(int) {}
};
static void feLog(const char*,double) { std::abort(); }
class AccelerateSparseSolver : public LinearSolver {
public:
  enum { FTSparseFactorizationCholesky=0, FTSparseFactorizationLDLT,
    FTSparseFactorizationLDLTUnpivoted, FTSparseFactorizationLDLTSBK,
    FTSparseFactorizationLDLTTPP, FTSparseFactorizationQR, FTSparseFactorizationCholeskyAtA };
  enum { OMSparseOrderAMD=0, OMSparseOrderMetis, OMSparseOrderCOLAMD };
  enum { ITSparseConjugateGradient=0, ITSparseGMRES, ITSparseDQGMRES, ITSparseFGMRES, ITSparseLSMR };
  class Implementation; Implementation* imp;
  explicit AccelerateSparseSolver(FEModel*); ~AccelerateSparseSolver();
  SparseMatrix* CreateSparseMatrix(Matrix_Type); bool SetSparseMatrix(SparseMatrix*);
  bool PreProcess(); bool Factor(); bool BackSolve(double*,double*); void Destroy();
  static void MyReportError(const char*) {} static void MyReportStatus(const char*) {}
  double condition_number() { std::abort(); }
};
#include "extracted-methods.inc"

static void attach(AccelerateSparseSolver& solver, CompactMatrix& matrix) {
  assert(solver.SetSparseMatrix(&matrix));
  solver.imp->m_ftype=AccelerateSparseSolver::FTSparseFactorizationLDLT;
  solver.imp->m_ordmthd=AccelerateSparseSolver::OMSparseOrderAMD;
}
static void clean(AccelerateSparseSolver& solver) {
  solver.Destroy(); solver.Destroy(); ledger::empty();
  assert(!solver.imp->m_ownsSymbolic && !solver.imp->m_ownsNumeric && !solver.imp->m_isFactored);
  assert(!solver.imp->ASS.factorization && !solver.imp->ASF.numericFactorization);
  assert(!solver.imp->ASF.symbolicFactorization.factorization && !solver.imp->colS);
}
static void scenario(int which) {
  using namespace ledger;
  reset(); CompactMatrix matrix; AccelerateSparseSolver solver(nullptr); attach(solver,matrix);
  double x[2]{},b[2]{};
  assert(!solver.BackSolve(x,b)); // nonzero, not factored: explicit false
  if(which==0) { assert(!solver.Factor()); assert(numeric_calls==0); clean(solver); }
  if(which==1) { assert(solver.PreProcess()); clean(solver); assert(symbolic_cleanups==1); }
  if(which==2) {
    assert(solver.PreProcess()); assert(solver.Factor()); assert(solver.BackSolve(x,b));
    assert(solver.Factor()); assert(numeric_calls==2 && numeric_cleanups==1 && solve_calls==1);
    clean(solver); assert(numeric_cleanups==2 && symbolic_cleanups==1);
  }
  if(which>=3 && which<=5) {
    assert(solver.PreProcess());
    numeric_mode=which==3 ? Numeric::failed_storage : which==4 ? Numeric::failed_symbolic_only : Numeric::failed_empty;
    assert(!solver.Factor()); assert(!solver.BackSolve(x,b));
    assert(numeric_storage.empty() && symbolic_refs.size()==1 && symbolic_refs.begin()->second==1);
    assert(numeric_cleanups==(which==5 ? 0 : 1));
    numeric_mode=Numeric::success; assert(solver.Factor()); clean(solver);
  }
  if(which==6) {
    assert(solver.PreProcess()); assert(solver.Factor()); numeric_mode=Numeric::failed_storage;
    assert(!solver.Factor()); assert(numeric_cleanups==2 && !solver.BackSolve(x,b)); clean(solver);
  }
  if(which==7 || which==8) {
    symbolic_mode=which==7 ? Symbolic::failed_owned : Symbolic::failed_empty;
    assert(!solver.PreProcess()); assert(!solver.Factor()); assert(!solver.BackSolve(x,b));
    assert(symbolic_cleanups==(which==7 ? 1 : 0)); clean(solver);
  }
  if(which==9) {
    for(int i=0;i<3;++i) { assert(solver.PreProcess()); assert(solver.Factor()); }
    assert(symbolic_cleanups==2 && numeric_cleanups==2); clean(solver);
  }
  if(which==10) {
    assert(solver.PreProcess()); assert(solver.Factor()); CompactMatrix replacement;
    attach(solver,replacement); empty(); assert(!solver.BackSolve(x,b));
    assert(solver.PreProcess()); assert(solver.Factor()); clean(solver);
    attach(solver,matrix);
  }
  if(which==11) {
    assert(solver.PreProcess()); assert(solver.Factor()); matrix.rows=matrix.columns=0;
    assert(solver.Factor()); empty(); int count=symbolic_calls+numeric_calls;
    assert(solver.PreProcess() && solver.Factor() && solver.BackSolve(x,b));
    assert(count==symbolic_calls+numeric_calls); clean(solver);
  }
  if(which==12) {
    assert(solver.PreProcess()); assert(solver.Factor());
    auto* created=solver.CreateSparseMatrix(REAL_SYMMETRIC); empty();
    assert(created && !solver.BackSolve(x,b)); assert(solver.PreProcess() && solver.Factor());
    clean(solver); delete created; attach(solver,matrix);
  }
  if(which==13) {
    {
      AccelerateSparseSolver nested(nullptr); attach(nested,matrix);
      assert(nested.PreProcess()); assert(nested.Factor());
      watched[0]=nested.imp; watched[1]=nested.imp->pointers.data(); released[0]=released[1]=false;
    }
    assert(released[0] && released[1]); watched[0]=watched[1]=nullptr; empty();
  }
  clean(solver);
  std::printf("PASS lifecycle scenario=%d\n",which);
}
int main() { for(int i=0;i<14;++i) scenario(i); ledger::empty(); return 0; }
