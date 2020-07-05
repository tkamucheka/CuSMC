// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

// MVN
Eigen::VectorXd MVN(Eigen::VectorXd mu, Eigen::MatrixXd sigma);
RcppExport SEXP _CuSMC_MVN(SEXP muSEXP, SEXP sigmaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type mu(muSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type sigma(sigmaSEXP);
    rcpp_result_gen = Rcpp::wrap(MVN(mu, sigma));
    return rcpp_result_gen;
END_RCPP
}
// MVNPDF
double MVNPDF(Eigen::VectorXd x, Eigen::VectorXd mu, Eigen::MatrixXd sigma);
RcppExport SEXP _CuSMC_MVNPDF(SEXP xSEXP, SEXP muSEXP, SEXP sigmaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type x(xSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type mu(muSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type sigma(sigmaSEXP);
    rcpp_result_gen = Rcpp::wrap(MVNPDF(x, mu, sigma));
    return rcpp_result_gen;
END_RCPP
}
// MVT
Eigen::VectorXd MVT(Eigen::VectorXd mu, Eigen::MatrixXd sigma, float nu);
RcppExport SEXP _CuSMC_MVT(SEXP muSEXP, SEXP sigmaSEXP, SEXP nuSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type mu(muSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< float >::type nu(nuSEXP);
    rcpp_result_gen = Rcpp::wrap(MVT(mu, sigma, nu));
    return rcpp_result_gen;
END_RCPP
}
// MVTPDF
double MVTPDF(Eigen::VectorXd x, Eigen::VectorXd mu, Eigen::MatrixXd sigma, float nu);
RcppExport SEXP _CuSMC_MVTPDF(SEXP xSEXP, SEXP muSEXP, SEXP sigmaSEXP, SEXP nuSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type x(xSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type mu(muSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< float >::type nu(nuSEXP);
    rcpp_result_gen = Rcpp::wrap(MVTPDF(x, mu, sigma, nu));
    return rcpp_result_gen;
END_RCPP
}
// run
List run(unsigned& N, unsigned& d, unsigned& timeSteps, Eigen::MatrixXd Y, Eigen::VectorXd m0, Eigen::MatrixXd C0, Eigen::MatrixXd F, float df, std::string resampler, std::string distribution);
RcppExport SEXP _CuSMC_run(SEXP NSEXP, SEXP dSEXP, SEXP timeStepsSEXP, SEXP YSEXP, SEXP m0SEXP, SEXP C0SEXP, SEXP FSEXP, SEXP dfSEXP, SEXP resamplerSEXP, SEXP distributionSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned& >::type N(NSEXP);
    Rcpp::traits::input_parameter< unsigned& >::type d(dSEXP);
    Rcpp::traits::input_parameter< unsigned& >::type timeSteps(timeStepsSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type Y(YSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type m0(m0SEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type C0(C0SEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type F(FSEXP);
    Rcpp::traits::input_parameter< float >::type df(dfSEXP);
    Rcpp::traits::input_parameter< std::string >::type resampler(resamplerSEXP);
    Rcpp::traits::input_parameter< std::string >::type distribution(distributionSEXP);
    rcpp_result_gen = Rcpp::wrap(run(N, d, timeSteps, Y, m0, C0, F, df, resampler, distribution));
    return rcpp_result_gen;
END_RCPP
}
// step
List step(unsigned& N, unsigned& d, unsigned& timeSteps, Eigen::MatrixXd Y, Eigen::MatrixXd F, float df, std::string resampler, std::string distribution);
RcppExport SEXP _CuSMC_step(SEXP NSEXP, SEXP dSEXP, SEXP timeStepsSEXP, SEXP YSEXP, SEXP FSEXP, SEXP dfSEXP, SEXP resamplerSEXP, SEXP distributionSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned& >::type N(NSEXP);
    Rcpp::traits::input_parameter< unsigned& >::type d(dSEXP);
    Rcpp::traits::input_parameter< unsigned& >::type timeSteps(timeStepsSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type Y(YSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type F(FSEXP);
    Rcpp::traits::input_parameter< float >::type df(dfSEXP);
    Rcpp::traits::input_parameter< std::string >::type resampler(resamplerSEXP);
    Rcpp::traits::input_parameter< std::string >::type distribution(distributionSEXP);
    rcpp_result_gen = Rcpp::wrap(step(N, d, timeSteps, Y, F, df, resampler, distribution));
    return rcpp_result_gen;
END_RCPP
}
// metropolis_hastings
Eigen::VectorXd metropolis_hastings(Eigen::VectorXd w, int N, int B);
RcppExport SEXP _CuSMC_metropolis_hastings(SEXP wSEXP, SEXP NSEXP, SEXP BSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type w(wSEXP);
    Rcpp::traits::input_parameter< int >::type N(NSEXP);
    Rcpp::traits::input_parameter< int >::type B(BSEXP);
    rcpp_result_gen = Rcpp::wrap(metropolis_hastings(w, N, B));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_CuSMC_MVN", (DL_FUNC) &_CuSMC_MVN, 2},
    {"_CuSMC_MVNPDF", (DL_FUNC) &_CuSMC_MVNPDF, 3},
    {"_CuSMC_MVT", (DL_FUNC) &_CuSMC_MVT, 3},
    {"_CuSMC_MVTPDF", (DL_FUNC) &_CuSMC_MVTPDF, 4},
    {"_CuSMC_run", (DL_FUNC) &_CuSMC_run, 10},
    {"_CuSMC_step", (DL_FUNC) &_CuSMC_step, 8},
    {"_CuSMC_metropolis_hastings", (DL_FUNC) &_CuSMC_metropolis_hastings, 3},
    {NULL, NULL, 0}
};

RcppExport void R_init_CuSMC(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
