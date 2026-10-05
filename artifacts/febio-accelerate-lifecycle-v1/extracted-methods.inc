class AccelerateSparseSolver::Implementation
{
public:
    FEModel*            m_fem;
    
    CompactMatrix*      m_pA;
    int                 m_mtype; // matrix type
    
#ifdef __APPLE__
    SparseMatrixStructure SMS;
    SparseMatrix_Double A;
    SparseOpaqueSymbolicFactorization ASS{};
    bool m_ownsSymbolic = false;
    SparseSymbolicFactorOptions SSFO;
    SparseOpaqueFactorization_Double ASF{};
    bool m_ownsNumeric = false;
    long* colS;
    std::vector<long> pointers;
    SparseIterativeStatus_t SStatus;
#endif
    
    // solver control parameters
    
    
    // Matrix data
    int m_n, m_nnz, m_nrhs;
    
    bool    m_print_cn;     // estimate and print the condition number
    bool    m_iparm3;       // use direct-iterative method
    bool    m_isFactored;
    int     m_ftype;        // factorization type
    int     m_ordmthd;      // order method
    int     m_itrmthd;      // iterative method
    int     m_maxiter;      // max number of iterations
    int     m_rtol;         // absolute tolerance
    int     m_atol;         // relative tolerance
    int     m_print_level;  // print level for status updates
    
public:
    Implementation()
    {
        m_print_cn = false;
        m_mtype = -2;
        m_iparm3 = false;
        m_isFactored = false;
        m_ftype = -1;
        m_ordmthd = -1;
        m_itrmthd = ITSparseLSMR;
        m_pA = nullptr;
        m_maxiter = 1000;
        m_rtol = 1e-3;
        m_atol = 1e-12;
        m_print_level = 0;
    }

    void ClearNumericFactor()
    {
        // A failed SDK result may still own numeric storage or a symbolic reference.
        if (m_ownsNumeric) SparseCleanup(ASF);
        ASF = {};
        m_ownsNumeric = false;
        m_isFactored = false;
    }
};

AccelerateSparseSolver::AccelerateSparseSolver(FEModel* fem) : LinearSolver(fem), imp(new AccelerateSparseSolver::Implementation)
{
    imp->m_fem = fem;
}

AccelerateSparseSolver::~AccelerateSparseSolver()
{
    Destroy();
    delete imp;
    imp = nullptr;
}

SparseMatrix* AccelerateSparseSolver::CreateSparseMatrix(Matrix_Type ntype)
{
    Destroy();
    // allocate the correct matrix format depending on matrix symmetry type
    switch (ntype)
    {
        case REAL_SYMMETRIC     : imp->m_mtype = -2; imp->m_pA = new CCSSparseMatrix(0); break;
        case REAL_UNSYMMETRIC   : imp->m_mtype = 11; imp->m_pA = new CCSSparseMatrix(0); break;
        case REAL_SYMM_STRUCTURE: imp->m_mtype =  1; imp->m_pA = new CCSSparseMatrix(0); break;
        default:
            assert(false);
            imp->m_pA = nullptr;
    }
    
    return imp->m_pA;
}

bool AccelerateSparseSolver::SetSparseMatrix(SparseMatrix* pA)
{
    Destroy();
    imp->m_pA = dynamic_cast<CompactMatrix*>(pA);
    imp->m_mtype = -2;
    if (dynamic_cast<CCSSparseMatrix*>(pA)) imp->m_mtype = 11;
    return (imp->m_pA != nullptr);
}

bool AccelerateSparseSolver::PreProcess()
{
    Destroy();
    
    imp->m_n = imp->m_pA->Rows();
    imp->m_nnz = imp->m_pA->NonZeroes();
    imp->m_nrhs = 1;
    
    if (imp->m_n == 0) return LinearSolver::PreProcess();

    const int pointerCount = imp->m_pA->Columns() + 1;
    imp->pointers.resize(pointerCount);
    for (int i=0; i<pointerCount; ++i) imp->pointers[i] = imp->m_pA->Pointers()[i];
    imp->colS = &imp->pointers[0];
    
    if (__builtin_available(macOS 10.13, *)) {
        imp->SMS = {};
        imp->SMS.attributes.triangle = SparseUpperTriangle;
        imp->SMS.columnCount = imp->m_pA->Columns();
        imp->SMS.rowCount = imp->m_pA->Rows();
        imp->SMS.rowIndices = imp->m_pA->Indices();
        imp->SMS.columnStarts = imp->colS;
        imp->SMS.blockSize = 1;
        if (!imp->m_iparm3) {
            imp->SSFO.control = SparseDefaultControl;
            imp->SSFO.order = nullptr;
            imp->SSFO.ignoreRowsAndColumns = nullptr;
            imp->SSFO.malloc = malloc;
            imp->SSFO.free = free;
            imp->SSFO.reportError = MyReportError;
            if (imp->m_mtype == 11) {
                // Use QR factorization as default for non-symmetric matrix
                imp->SMS.attributes.kind = SparseOrdinary;
                switch (imp->m_ordmthd) {
                    case OMSparseOrderAMD:
                        imp->SSFO.orderMethod = SparseOrderAMD;
                        break;
                    case OMSparseOrderMetis:
                        imp->SSFO.orderMethod = SparseOrderMetis;
                        break;
                    case OMSparseOrderCOLAMD:
                        imp->SSFO.orderMethod = SparseOrderCOLAMD;
                        break;
                    default:
                        imp->SSFO.orderMethod = SparseOrderCOLAMD;
                        break;
                }
                switch (imp->m_ftype) {
                    case -1:
                    case FTSparseFactorizationQR:
                        imp->ASS = SparseFactor(SparseFactorizationQR, imp->SMS, imp->SSFO);
                        break;
                    case FTSparseFactorizationCholeskyAtA:
                        imp->ASS = SparseFactor(SparseFactorizationCholeskyAtA, imp->SMS, imp->SSFO);
                        break;
                    default:
                        fprintf(stderr, "\nIncorrect factorization type for non-symmetric matrix!");
                        return false;
                        break;
                }
            }
            else {
                // Use LDLT as default factorization for symmetric matrix
                // (Cholesky factorization does not seem to work)
                imp->SMS.attributes.kind = SparseSymmetric;
                switch (imp->m_ordmthd) {
                    case -1:
                    case OMSparseOrderAMD:
                        imp->SSFO.orderMethod = SparseOrderAMD;
                        break;
                    case OMSparseOrderMetis:
                        imp->SSFO.orderMethod = SparseOrderMetis;
                        break;
                    default:
                        fprintf(stderr, "\nIncorrect order method for symmetric matrix!");
                        return false;
                        break;
                }
                switch (imp->m_ftype) {
                    case FTSparseFactorizationCholesky:
                        imp->ASS = SparseFactor(SparseFactorizationCholesky, imp->SMS, imp->SSFO);
                        break;
                    case FTSparseFactorizationLDLT:
                        imp->ASS = SparseFactor(SparseFactorizationLDLT, imp->SMS, imp->SSFO);
                        break;
                    case FTSparseFactorizationLDLTUnpivoted:
                        imp->ASS = SparseFactor(SparseFactorizationLDLTUnpivoted, imp->SMS, imp->SSFO);
                        break;
                    case FTSparseFactorizationLDLTSBK:
                        imp->ASS = SparseFactor(SparseFactorizationLDLTSBK, imp->SMS, imp->SSFO);
                        break;
                    case -1:    // default factorization
                    case FTSparseFactorizationLDLTTPP:
                        imp->ASS = SparseFactor(SparseFactorizationLDLTTPP, imp->SMS, imp->SSFO);
                        break;
                    default:
                        fprintf(stderr, "\nIncorrect factorization type for symmetric matrix!");
                        return false;
                        break;
                }
            }
            imp->m_ownsSymbolic = (imp->ASS.factorization != nullptr);
            if (imp->ASS.status != SparseStatusOK || !imp->m_ownsSymbolic) {
                Destroy();
                return false;
            }
        }
        else {
            // for iterative solving we don't need to factorize
            return true;
        }
    }
    else {
        // Fallback on earlier versions
        fprintf(stderr, "\nERROR during preprocessing: ");
        fprintf(stderr, "\naccelerate solver not available for macOS earlier than 10.13!");
        return false;
    }
    
    return LinearSolver::PreProcess();
}

bool AccelerateSparseSolver::Factor()
{
    // Discard the previous numeric factor before allocating a fresh one.
    imp->ClearNumericFactor();
    // make sure we have work to do
    if (imp->m_pA->Rows() == 0) { Destroy(); return true; }
    if (!imp->m_iparm3 && !imp->m_ownsSymbolic) return false;
    
    // ------------------------------------------------------------------------------
    // This step does the factorization
    // ------------------------------------------------------------------------------

    if (__builtin_available(macOS 10.13, *)) {

        // create the sparse matrix
        imp->A.data = imp->m_pA->Values();
        imp->A.structure = imp->SMS;

        // for iterative solves we don't factor
        if (imp->m_iparm3) {
            return true;
        }
        else {
            imp->ASF = SparseFactor(imp->ASS,imp->A);
            imp->m_ownsNumeric = (imp->ASF.symbolicFactorization.status >= 0 &&
                (imp->ASF.numericFactorization != nullptr ||
                 imp->ASF.symbolicFactorization.factorization != nullptr));
            if (imp->ASF.status != SparseStatusOK) {
                fprintf(stderr, "\n\tERROR during factorization:");
                switch (imp->ASF.status) {
                    case SparseFactorizationFailed:
                        fprintf(stderr, "\n\tFactorization failed!");
                        break;
                    case SparseMatrixIsSingular:
                        fprintf(stderr, "\n\tSparse matrix is singular!");
                        break;
                    case SparseInternalError:
                        fprintf(stderr, "\n\tSolver called internal error!");
                        break;
                    case SparseParameterError:
                        fprintf(stderr, "\n\tSolver called parameter error!");
                        break;
                    default:
                        fprintf(stderr, "\n\tUnknown error!");
                }
                imp->ClearNumericFactor();
                return false;
            }
        }
    }
    else {
        // Fallback on earlier versions
        fprintf(stderr, "\nERROR during factorization: ");
        fprintf(stderr, "\nAccelerate solver not available for macOS earlier than 10.13!");
        return false;
    }
    
    // calculate and print the condition number
    if (imp->m_print_cn)
    {
        double c = condition_number();
        feLog("\tcondition number (est.) ................... : %lg\n\n", c);
    }
    
    imp->m_isFactored = true;
    
    return true;
}

bool AccelerateSparseSolver::BackSolve(double* x, double* b)
{
    // make sure we have work to do
    if (imp->m_pA->Rows() == 0) return true;
    if ((!imp->m_iparm3) && (imp->m_isFactored == false)) return false;
    
    DenseVector_Double X, B;
    X.count = B.count = imp->m_n;
    X.data = x; B.data = b;
    
    // solve the system
    if (__builtin_available(macOS 10.13, *)) {
        if (!imp->m_iparm3) {
            SparseSolve(imp->ASF,B,X);
        }
        else {
            switch (imp->m_itrmthd) {
                case ITSparseConjugateGradient:
                    SparseCGOptions cg_opt;
                    cg_opt.maxIterations = imp->m_maxiter;
                    cg_opt.rtol = imp->m_rtol;
                    cg_opt.atol = imp->m_atol;
                    cg_opt.reportError = MyReportError;
                    cg_opt.reportStatus = (imp->m_print_level>0) ? MyReportStatus : nullptr;
                    imp->SStatus = SparseSolve(SparseConjugateGradient(cg_opt),imp->A,B,X,SparsePreconditionerDiagonal);
                    break;
                case ITSparseGMRES:
                case ITSparseDQGMRES:
                case ITSparseFGMRES:
                    SparseGMRESOptions gmres_opt;
                    gmres_opt.maxIterations = imp->m_maxiter;
                    gmres_opt.reportError = MyReportError;
                    gmres_opt.reportStatus = (imp->m_print_level>0) ? MyReportStatus : nullptr;
                    gmres_opt.nvec = 0;
                    gmres_opt.rtol = imp->m_rtol;
                    gmres_opt.atol = imp->m_atol;
                    switch (imp->m_itrmthd) {
                        case ITSparseGMRES:
                            gmres_opt.variant = SparseVariantGMRES;
                            imp->SStatus = SparseSolve(SparseGMRES(gmres_opt),imp->A,B,X,SparsePreconditionerDiagonal);
                            break;
                        case ITSparseDQGMRES:
                            gmres_opt.variant = SparseVariantDQGMRES;
                            imp->SStatus = SparseSolve(SparseGMRES(gmres_opt),imp->A,B,X,SparsePreconditionerNone);
                            break;
                        case ITSparseFGMRES:
                            gmres_opt.variant = SparseVariantFGMRES;
                            imp->SStatus = SparseSolve(SparseGMRES(gmres_opt),imp->A,B,X,SparsePreconditionerDiagScaling);
                            break;
                    }
                    break;
                case ITSparseLSMR:
                    SparseLSMROptions lsmr_opt;
                    lsmr_opt.rtol = imp->m_rtol;
                    lsmr_opt.atol = imp->m_atol;
                    lsmr_opt.reportError = MyReportError;
                    lsmr_opt.reportStatus = (imp->m_print_level>0) ? MyReportStatus : nullptr;
                    imp->SStatus = SparseSolve(SparseLSMR(lsmr_opt),imp->A,B,X,SparsePreconditionerDiagScaling);
                    break;
            }
            if (imp->SStatus != SparseIterativeConverged) {
                fprintf(stderr, "\n\tERROR during iterative solution:\n");
                switch (imp->SStatus) {
                    case SparseIterativeIllConditioned:
                        fprintf(stderr, "\n\tIll-conditioned system!\n");
                        break;
                    case SparseIterativeInternalError:
                        fprintf(stderr, "\n\tInternal failure!\n");
                        break;
                    case SparseIterativeMaxIterations:
                        fprintf(stderr, "\n\tExceeded maximum iteration limit!\n");
                        break;
                    case SparseIterativeParameterError:
                        fprintf(stderr, "\n\tError with one or more parameters!\n");
                        break;
                    default:
                        fprintf(stderr, "\n\tUnknown error!\n");
                }
                return false;
            }
        }
    } else {
        // Fallback on earlier versions
        fprintf(stderr, "\nERROR during back solve: ");
        fprintf(stderr, "\nAccelerate solver not available for macOS earlier than 10.13!");
        return false;
    }
    
    // update stats
    UpdateStats(1);
    
    return true;
}

void AccelerateSparseSolver::Destroy()
{
    if (imp == nullptr) return;
    imp->ClearNumericFactor();
    if (imp->m_ownsSymbolic) SparseCleanup(imp->ASS);
    imp->ASS = {};
    imp->m_ownsSymbolic = false;
    imp->A = {};
    imp->SMS = {};
    imp->colS = nullptr;
    imp->pointers.clear();
}
