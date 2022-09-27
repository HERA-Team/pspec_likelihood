pspec_likelihood
================

**A small but powerful interface to generate theoretical likelihoods from ``UVPSpec`` objects.**


.. image:: https://github.com/hera-team/pspec_likelihood/workflows/Tests/badge.svg
    :target: https://github.com/steven-murray/hmf
.. image:: https://codecov.io/gh/hera-team/pspec_likelihood/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/steven-murray/hmf
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
.. image:: https://results.pre-commit.ci/badge/github/HERA-Team/pspec_likelihood/main.svg
   :target: https://results.pre-commit.ci/latest/github/HERA-Team/pspec_likelihood/main
   :alt: pre-commit.ci status

Full Documentation
------------------
`Read the docs. <http://pspec_likelihood.readthedocs.org>`_

Features
--------
* Ingests data output from ``hera_pspec``: power spectra, covariance matrices and
  window functions.
* Agnostic to theory code (i.e. run 21cmFAST or ARES or any other model)
* Outputs a log-likelihood to be used in parameter inference, but is sampler
  agnostic.


Installation
------------
Clone/download the repo and ``pip install .``, or ``pip install git+git://github.com/hera-team/pspec_likelihood``.

If developing::

    git clone https://github.com/hera-team/pspec_likelihood
    cd pspec_likelihood
    pip install -e .[dev]
    pre-commit install


Quickstart
----------
Import like this::

    from pspec_likelihood import DataModelInterface, Gaussian

To construct a likelihood, you first need to construct the ``DataModelInterface``,
for which you will specify the data, its covariance, a window function,
and a model both for the theory and the systematics. This class contains all the methods
required to compute the model/systematics and transform it consistently to data-space.

Secondly, you need to construct a ``PSpecLikelihood``, via one of its concrete sub-classes.
Examples of such subclasses are ``Gaussian`` and ``GaussianLinearSystematics``. The
reason these are their own class, instead of being part of the ``DataModelInterface``,
is for the sake of modularity and extensibility. This allows different actual likelihoods
to be computed given the data, and new likelihoods to be implemented with ease.
The basic requirement of a ``PSpecLikelihood`` subclass is that it must implement the
``loglike(theory_params, sys_params)`` method, which goes and computes the actual
log-likelihood given a set of parameters. It has access to the ``DataModelInterface``
object through its ``model`` attribute. So, eg.::

    likelihood = Gaussian(
        model = DataModelInterface(...)
    )

    likelihood.loglike(theory_params, sys_params)

Versioning
----------
From v0.1.0, ``pspec_likelihood`` will be using strict semantic versioning, such that increases in
the **major** version have potential API breaking changes, **minor** versions introduce
new features, and **patch** versions fix bugs and other non-breaking internal changes.

If your package depends on ``pspec_likelihood``, set the dependent version like this::

    pspec_likelihood>=0.1
