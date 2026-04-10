/*
 * cffi_unuran.h — UNU.RAN CFFI interface declarations
 *
 * Minimal C declarations consumed by the CFFI build script
 * (_cffi_build.py) to generate the Python bindings for the UNU.RAN
 * random-variate generation library.
 *
 * Covers:
 *   - Opaque handle typedefs (UNUR_DISTR, UNUR_GEN, UNUR_PAR, UNUR_URNG)
 *   - Distribution constructors and setter functions (continuous & discrete)
 *   - Method (algorithm) constructors: AROU, TDR, HINV, PINV, NINV, DGT
 *   - Generator lifecycle: init, sample, free
 *   - Custom URNG registration
 *   - Error query helpers
 *
 * Author: Artem Romanyuk
 * Copyright (c) 2025 PySATL project
 * SPDX-License-Identifier: MIT
 */

struct unur_distr;
struct unur_gen;
struct unur_par;
struct unur_urng;

typedef struct unur_distr* UNUR_DISTR;
typedef struct unur_gen* UNUR_GEN;
typedef struct unur_par* UNUR_PAR;
typedef struct unur_urng* UNUR_URNG;

UNUR_DISTR unur_distr_cont_new(void);
UNUR_DISTR unur_distr_discr_new(void);

int unur_distr_cont_set_pdf(UNUR_DISTR distribution,
                            double (*pdf)(double, const struct unur_distr*));
int unur_distr_cont_set_dpdf(UNUR_DISTR distribution,
                            double (*dpdf)(double, const struct unur_distr*));
int unur_distr_cont_set_cdf(UNUR_DISTR distribution,
                            double (*cdf)(double, const struct unur_distr*));
int unur_distr_cont_set_invcdf(UNUR_DISTR distribution,
                                double (*invcdf)(double, const struct unur_distr*));
int unur_distr_cont_set_domain(UNUR_DISTR distribution, double left, double right);
int unur_distr_cont_set_mode(UNUR_DISTR distribution, double mode);
int unur_distr_cont_set_pdfparams(UNUR_DISTR distribution, const double* params, int n_params);

int unur_distr_discr_set_pmf(UNUR_DISTR distribution,
                            double (*pmf)(int, const struct unur_distr*));
int unur_distr_discr_set_cdf(UNUR_DISTR distribution,
                            double (*cdf)(int, const struct unur_distr*));
int unur_distr_discr_set_pv(UNUR_DISTR distribution, const double* pv, int n_pv);
int unur_distr_discr_set_pmfparams(UNUR_DISTR distribution, const double* params, int n_params);
int unur_distr_discr_set_domain(UNUR_DISTR distribution, int left, int right);
int unur_distr_discr_set_pmfsum(UNUR_DISTR distribution, double sum);
int unur_distr_discr_make_pv(UNUR_DISTR distribution);

UNUR_PAR unur_arou_new(const UNUR_DISTR distribution);
UNUR_PAR unur_tdr_new(const UNUR_DISTR distribution);
UNUR_PAR unur_hinv_new(const UNUR_DISTR distribution);
UNUR_PAR unur_pinv_new(const UNUR_DISTR distribution);
UNUR_PAR unur_ninv_new(const UNUR_DISTR distribution);
UNUR_PAR unur_dgt_new(const UNUR_DISTR distribution);

UNUR_GEN unur_init(UNUR_PAR parameters);

UNUR_URNG unur_urng_new(double (*sampleunif)(void *state), void *state);
UNUR_URNG unur_set_default_urng(UNUR_URNG urng_new);
UNUR_URNG unur_set_default_urng_aux(UNUR_URNG urng_new);
void unur_urng_free(UNUR_URNG urng);

double unur_sample_cont(UNUR_GEN generator);
int unur_sample_discr(UNUR_GEN generator);
int unur_sample_vec(UNUR_GEN generator, double* vector);

double unur_quantile(UNUR_GEN generator, double U);

void unur_free(UNUR_GEN generator);
void unur_distr_free(UNUR_DISTR distribution);
void unur_par_free(UNUR_PAR par);

const char* unur_get_strerror(const int errnocode);
int unur_get_errno(void);
const char* unur_gen_info(UNUR_GEN generator, int help);
