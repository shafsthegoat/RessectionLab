// Repaired-source-extracted initialization only. No SparseFactor/SparseSolve calls.
#include <Accelerate/Accelerate.h>
#include <cassert>
#include <cstdio>
struct Matrix {
    int n;
    int* indices;
    int Rows() const { return n; }
    int Columns() const { return n; }
    int* Indices() const { return indices; }
};
struct State {
    SparseMatrixStructure SMS{};
    Matrix* m_pA = nullptr;
    long* colS = nullptr;
};
void initialize_repaired(State* imp, bool symmetric) {
        imp->SMS = {};
        imp->SMS.attributes.triangle = SparseUpperTriangle;
        imp->SMS.columnCount = imp->m_pA->Columns();
        imp->SMS.rowCount = imp->m_pA->Rows();
        imp->SMS.rowIndices = imp->m_pA->Indices();
        imp->SMS.columnStarts = imp->colS;
        imp->SMS.blockSize = 1;
    if (symmetric) {
                imp->SMS.attributes.kind = SparseSymmetric;
    } else {
                imp->SMS.attributes.kind = SparseOrdinary;
    }
}
void set_defined_stale_fields(State& state, long* starts, int* indices) {
    // Every field is defined before the repaired fragment. These intentionally
    // nondefault metadata values are never handed to a Sparse API function.
    state.SMS.rowCount = 17;
    state.SMS.columnCount = 19;
    state.SMS.columnStarts = starts;
    state.SMS.rowIndices = indices;
    state.SMS.blockSize = 7;
    state.SMS.attributes.transpose = true;
    state.SMS.attributes.triangle = SparseLowerTriangle;
    state.SMS.attributes.kind = SparseUnitTriangular;
    state.SMS.attributes._reserved = 2047;
    state.SMS.attributes._allocatedBySparse = true;
}
int main() {
    static_assert(SparseUpperTriangle == 0, "Review the pinned SDK convention");
    long starts2[] = {0, 2, 4};
    int rows2[] = {0, 1, 0, 1};
    long starts3[] = {0, 3, 6, 9};
    int rows3[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    long stale_starts[] = {0, 1};
    int stale_rows[] = {0};
    Matrix matrices[] = {{2, rows2}, {3, rows3}};
    State state{};
    for (int profile = 0; profile < 4; ++profile) {
        const int id = profile / 2;
        const bool symmetric = profile % 2 == 0;
        set_defined_stale_fields(state, stale_starts, stale_rows);
        state.m_pA = &matrices[id];
        state.colS = id == 0 ? starts2 : starts3;
        initialize_repaired(&state, symmetric);
        assert(state.SMS.rowCount == matrices[id].n);
        assert(state.SMS.columnCount == matrices[id].n);
        assert(state.SMS.rowIndices == matrices[id].indices);
        assert(state.SMS.columnStarts == state.colS);
        assert(state.SMS.blockSize == 1);
        assert(state.SMS.attributes.transpose == false);
        assert(state.SMS.attributes.triangle == SparseUpperTriangle);
        assert(state.SMS.attributes.kind == (symmetric ? SparseSymmetric : SparseOrdinary));
        assert(state.SMS.attributes._reserved == 0);
        assert(state.SMS.attributes._allocatedBySparse == false);
        std::printf("PASS matrix-init profile=%d rows=%d symmetric=%d\n",
                    profile, matrices[id].n, static_cast<int>(symmetric));
    }
}
