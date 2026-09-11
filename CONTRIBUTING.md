# Contributing to A-MESH

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

## Types of Contributions

### Report Bugs

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation

You can never have enough documentation! Please feel free to contribute to any
part of the documentation, such as the official docs, docstrings, or even
on the web in blog posts, articles, and such.

### Submit Feedback

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## Get Started!

Ready to contribute? Follow these steps to set up A-MESH for local development.

1. Clone the repository and enter its directory:

    ```console
    git clone https://github.com/LeoVAF/AMesh-python.git AMesh-python
    cd AMesh-python
    ```

2. Create and activate the complete Conda environment:

    ```console
    conda env create --file environment.yml
    conda activate amesh
    ```

   This installs Python 3.11, the runtime dependencies, development tools, and
   the `amesh` package in editable mode.

3. Create a branch for your changes:

    ```console
    git checkout -b name-of-your-bugfix-or-feature
    ```

4. Run the test suite:

    ```console
    pytest
    ```

5. If documentation changed, build the HTML site and check the generated pages:

    ```console
    make -C docs html
    ```

   The entry point is `docs/_build/html/index.html`.

6. Commit your changes and open a pull request.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include additional tests if appropriate.
2. If the pull request adds functionality, the docs should be updated.
3. The test suite should pass under the supported Python 3.11 environment.
4. Public-facing prose should use **A-MESH**. Literal Python identifiers, such
   as `amesh`, `AMESH`, and `AMESHParameters`, must match the implemented API.

## Code of Conduct

Please note that the A-MESH project is released with a
Code of Conduct. By contributing to this project you agree to abide by its terms.
