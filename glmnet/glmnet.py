import numpy as np
import _glmnet

_DEFAULT_THRESH = 1.0e-4
_DEFAULT_FLMIN = 0.001
_DEFAULT_NLAM = 100


def elastic_net(X, y, rho, pos=True, thr=1.0e-4, weights=None, vp=None,
                isd=True, nlam=100, maxit=1000, intr=False, **kwargs):
    """
    Raw-output wrapper for elastic net linear regression.
    """

    # Mandatory parameters
    X = np.asanyarray(X)
    y = np.asanyarray(y)

    if y.ndim != 2:
        y.shape = (y.shape + (1,))
    # print(X.shape)
    memlimit = X.shape[1]

    # # Flags determining overwrite behavior
    # overwrite_pred_ok = False
    # overwrite_targ_ok = False

    # thr = 1.0e-4   # Minimum change in largest coefficient
    # weights = None          # Relative weighting per observation case
    # vp = None               # Relative penalties per predictor (0 = no penalty)
    # isd = True              # Standardize input variables before proceeding?
    jd = np.zeros(1)        # X to exclude altogether from fitting
    ulam = None             # User-specified lambda values
    flmin = 0.001  # Fraction of largest lambda at which to stop
    # nlam = 100    # The (maximum) number of lambdas to try.
    # maxit = 1000

    box_constraints = np.zeros((2, X.shape[1]), order='F')
    box_constraints[1] = 1e300

    if not pos:
        box_constraints[0] = -1e300

    for keyword in kwargs:
        # if keyword == 'overwrite_pred_ok':
        #     overwrite_pred_ok = kwargs[keyword]
        # elif keyword == 'overwrite_targ_ok':
        #     overwrite_targ_ok = kwargs[keyword]
        # if keyword == 'threshold':
        #     thr = kwargs[keyword]
        # elif keyword == 'weights':
        #     weights = np.asarray(kwargs[keyword]).copy()
        # elif keyword == 'penalties':
        #     vp = kwargs[keyword].copy()
        # elif keyword == 'standardize':
        #     isd = bool(kwargs[keyword])
        if keyword == 'exclude':
            # Add one since Fortran indices start at 1
            exclude = (np.asarray(kwargs[keyword]) + 1).tolist()
            jd = np.array([len(exclude)] + exclude)
        elif keyword == 'lambdas':
            if 'flmin' in kwargs:
                raise ValueError("Can't specify both lambdas & flmin keywords")
            ulam = np.asarray(kwargs[keyword])
            flmin = 2.  # Pass flmin > 1.0 indicating to use the user-supplied.
            nlam = len(ulam)
        elif keyword == 'flmin':
            flmin = kwargs[keyword]
            ulam = None
        elif keyword == 'nlam':
            if 'lambdas' in kwargs:
                raise ValueError("Can't specify both lambdas & nlam keywords")
            nlam = kwargs[keyword]
        else:
            raise ValueError("Unknown keyword argument '%s'" % keyword)

    # # If X is a Fortran contiguous array, it will be overwritten.
    # # Decide whether we want this. If it's not Fortran contiguous it will
    # # be copied into that form anyway so there's no chance of overwriting.
    # if np.isfortran(X):
    #     if not overwrite_pred_ok:
    #         # Might as well make it F-ordered to avoid ANOTHER copy.
    #         X = X.copy(order='F')

    # y being a 1-dimensional array will usually be overwritten
    # with the standardized version unless we take steps to copy it.
    # if not overwrite_targ_ok:
    #     y = y.copy()

    # Uniform weighting if no weights are specified.
    if weights is None:
        weights = np.ones(X.shape[0])
    else:
        weights = np.asarray(weights).copy()
    # Uniform penalties if none were specified.
    if vp is None:
        vp = np.ones(X.shape[1])
    else:
        vp = vp.copy()

    # Call the Fortran wrapper.

    nx = X.shape[1]
    ny = y.shape[1]

    a0 = np.zeros((ny, nlam), dtype=np.float64)
    ca = np.zeros((ny, nx, nlam), dtype=np.float64)
    # ca_ = np.zeros((nx, nlam), dtype=np.float64)
    ia = np.zeros((ny, nx), dtype=np.int32)
    nin = np.zeros((ny, nlam), dtype=np.int32)
    alm = np.zeros((nlam), dtype=np.float64)
    # print(a0.shape, ca.shape, X.shape, y.shape)
    for idx in range(y.shape[1]):

        # X/y is overwritten in the fortran function at every loop, so we must copy it each time
        X_copy = X.copy(order='F')
        y_copy = y[:, idx].copy(order='F')
        # print(X_copy.sum(), y_copy.sum(), jd.sum(), box_constraints.sum(), X.sum(), y.sum())
        lmu, a0[idx], ca[idx], ia[idx], nin[idx], rsq, alm[:], nlp, jerr = \
            _glmnet.elnet(rho, X_copy, y_copy, weights, jd, vp, box_constraints, memlimit, flmin, ulam, thr,
                  nlam=nlam, isd=isd, maxit=maxit, intr=intr)

        # print(y.shape, X.shape, ca[idx].shape, np.sum(ca[idx] != 0), ia[idx].shape, nin[idx])
        # 1/0
        # get list of coefficient in right order
        # ia[idx, :nin] -= 1
        # print(ia)
        # print(ca_)
        # print(nin)
        # ca[idx] = ca_[:, :nin]
        # ca[idx] = solns(X.shape[0], X.shape[1], lmu, ca[idx], ia[idx], nin[idx])

    # print(X_copy.sum(), y_copy.sum(), jd.sum(), box_constraints.sum(), X.sum(), y.sum())
    # # Check for errors, documented in glmnet.f.
    # if jerr != 0:
    #     if jerr == 10000:
    #         raise ValueError('cannot have max(vp) < 0.0')
    #     elif jerr == 7777:
    #         raise ValueError('all used X have 0 variance')
    #     elif jerr < 7777:
    #         raise MemoryError('elnet() returned error code %d' % jerr)
    #     else:
    #         raise Exception('unknown error: %d' % jerr)



    # We substract 1 for the indexes since fortran indices start at 1
    # and python at 0
    # ia -= 1
    # c
    # ia = np.trim_zeros(ia, 'b') - 1
    # print(ia.shape)
    return lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr


# def elastic_net(predictors, target, balance, memlimit=None,
#                 largest=None, **kwargs):
#     """
#     Raw-output wrapper for elastic net linear regression.
#     """

#     # Mandatory parameters
#     predictors = np.asanyarray(predictors)
#     target = np.asanyarray(target)

#     # Decide on largest allowable models for memory/convergence.
#     memlimit = predictors.shape[1] if memlimit is None else memlimit

#     # If largest isn't specified use memlimit.
#     largest = memlimit if largest is None else largest

#     if memlimit < largest:
#         raise ValueError('Need largest <= memlimit')

#     # Flags determining overwrite behavior
#     overwrite_pred_ok = False
#     overwrite_targ_ok = False

#     thr = _DEFAULT_THRESH   # Minimum change in largest coefficient
#     weights = None          # Relative weighting per observation case
#     vp = None               # Relative penalties per predictor (0 = no penalty)
#     isd = True              # Standardize input variables before proceeding?
#     jd = np.zeros(1)        # Predictors to exclude altogether from fitting
#     ulam = None             # User-specified lambda values
#     flmin = _DEFAULT_FLMIN  # Fraction of largest lambda at which to stop
#     nlam = _DEFAULT_NLAM    # The (maximum) number of lambdas to try.

#     for keyword in kwargs:
#         if keyword == 'overwrite_pred_ok':
#             overwrite_pred_ok = kwargs[keyword]
#         elif keyword == 'overwrite_targ_ok':
#             overwrite_targ_ok = kwargs[keyword]
#         elif keyword == 'threshold':
#             thr = kwargs[keyword]
#         elif keyword == 'weights':
#             weights = np.asarray(kwargs[keyword]).copy()
#         elif keyword == 'penalties':
#             vp = kwargs[keyword].copy()
#         elif keyword == 'standardize':
#             isd = bool(kwargs[keyword])
#         elif keyword == 'exclude':
#             # Add one since Fortran indices start at 1
#             exclude = (np.asarray(kwargs[keyword]) + 1).tolist()
#             jd = np.array([len(exclude)] + exclude)
#         elif keyword == 'lambdas':
#             if 'flmin' in kwargs:
#                 raise ValueError("Can't specify both lambdas & flmin keywords")
#             ulam = np.asarray(kwargs[keyword])
#             flmin = 2. # Pass flmin > 1.0 indicating to use the user-supplied.
#             nlam = len(ulam)
#         elif keyword == 'flmin':
#             flmin = kwargs[keyword]
#             ulam = None
#         elif keyword == 'nlam':
#             if 'lambdas' in kwargs:
#                 raise ValueError("Can't specify both lambdas & nlam keywords")
#             nlam = kwargs[keyword]
#         else:
#             raise ValueError("Unknown keyword argument '%s'" % keyword)

#     # If predictors is a Fortran contiguous array, it will be overwritten.
#     # Decide whether we want this. If it's not Fortran contiguous it will
#     # be copied into that form anyway so there's no chance of overwriting.
#     if np.isfortran(predictors):
#         if not overwrite_pred_ok:
#             # Might as well make it F-ordered to avoid ANOTHER copy.
#             predictors = predictors.copy(order='F')

#     # target being a 1-dimensional array will usually be overwritten
#     # with the standardized version unless we take steps to copy it.
#     if not overwrite_targ_ok:
#         target = target.copy()

#     # Uniform weighting if no weights are specified.
#     if weights is None:
#         weights = np.ones(predictors.shape[0])

#     # Uniform penalties if none were specified.
#     if vp is None:
#         vp = np.ones(predictors.shape[1])

#     # Call the Fortran wrapper.
#     lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr =  \
#             _glmnet.elnet(balance, predictors, target, weights, jd, vp,
#                           memlimit, flmin, ulam, thr, nlam=nlam)

#     # Check for errors, documented in glmnet.f.
#     if jerr != 0:
#         if jerr == 10000:
#             raise ValueError('cannot have max(vp) < 0.0')
#         elif jerr == 7777:
#             raise ValueError('all used predictors have 0 variance')
#         elif jerr < 7777:
#             raise MemoryError('elnet() returned error code %d' % jerr)
#         else:
#             raise Exception('unknown error: %d' % jerr)

#     return lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr

class GlmnetLinearModel(object):
    """Class representing a linear model trained by Glmnet."""
    def __init__(self, a0, ca, ia, nin, rsq, alm, npred):
        self._intercept = a0
        self._coefficients = ca[:nin]
        self._indices = ia[:nin] - 1
        self._rsquared = rsq
        self._lambda = alm
        self._npred = npred

    def __str__(self):
        return ("%s with %d non-zero coefficients (%.2f%%)\n" + \
                " * Intercept = %.7f, Lambda = %.7f\n" + \
                " * Training r^2: %.4f") % \
                (self.__class__.__name__, len(self._coefficients),
                 len(self._coefficients) / float(self._npred) * 100,
                 self._intercept, self._lambda, self._rsquared)

    def predict(self, predictors):
        predictors = np.atleast_2d(np.asarray(predictors))
        return self._intercept + \
                np.dot(predictors[:,self._indices], self._coefficients)

    @property
    def coefficients(self):
        coeffs = np.zeros(self._npred)
        coeffs[self._indices] = self._coefficients
        return coeffs



class GlmnetLinearResults(object):
    def __init__(self, lmu, a0, ca, ia, nin, rsq, alm, nlp, npred, parm):
        self._lmu = lmu
        self._a0 = a0
        self._ca = ca
        self._ia = ia
        self._nin = nin
        self._rsq = rsq
        self._alm = alm
        self._nlp = nlp
        self._npred = npred
        self._model_objects = {}
        self._parm = parm

    def __str__(self):
        ninp = np.argmax(self._nin)
        return ("%s object, elastic net parameter = %.3f\n" + \
                " * %d values of lambda\n" + \
            " * computed in %d passes over data\n" + \
            " * largest model: %d predictors (%.1f%%), train r^2 = %.4f") % \
            (self.__class__.__name__, self._parm, self._lmu, self._nlp,
             self._nin[ninp], self._nin[ninp] / float(self._npred) * 100,
             self._rsq[ninp])

    def __len__(self):
        return self._lmu

    def __getitem__(self, item):
        item = (item + self._lmu) if item < 0 else item
        if item >= self._lmu or item < 0:
            raise IndexError("model index out of bounds")

        if item not in self._model_objects:
            model =  GlmnetLinearModel(
                        self._a0[item],
                        self._ca[:,item],
                        self._ia,
                        self._nin[item],
                        self._rsq[item],
                        self._alm[item],
                        self._npred
                    )
            self._model_objects[item] = model

        else:
            model = self._model_objects[item]

        return model

    @property
    def nummodels(self):
        return self._lmu

    @property
    def coefficients(self):
        return self._ca[:np.max(self._nin), :self._lmu]

    @property
    def indices(self):
        return self._ia

    @property
    def lambdas(self):
        return self._alm[:self._lmu]

    @property
    def balance(self):
        return self._parm

def plot_paths(results, which_to_label=None):
    import matplotlib
    import matplotlib.pyplot as plt
    plt.clf()
    interactive_state = plt.isinteractive()
    xvalues = -np.log(results.lambdas[1:])
    for index, path in enumerate(results.coefficients):
        if which_to_label and results.indices[index] in which_to_label:
            if which_to_label[results.indices[index]] is None:
                label = "$x_{%d}$" % results.indices[index]
            else:
                label = which_to_label[results.indices[index]]
        else:
            label = None


        if which_to_label and label is None:
            plt.plot(xvalues, path[1:], ':')
        else:
            plt.plot(xvalues, path[1:], label=label)

    plt.xlim(np.amin(xvalues), np.amax(xvalues))

    if which_to_label is not None:
        plt.legend(loc='upper left')
    plt.title('Regularization paths ($\\rho$ = %.2f)' % results.balance)
    plt.xlabel('$-\log(\lambda)$')
    plt.ylabel('Value of regression coefficient $\hat{\\beta}_i$')
    plt.show()
    plt.interactive(interactive_state)
