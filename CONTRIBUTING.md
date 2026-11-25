# PySATL Core project contributing guide

Thank you very much if you have decided to contribute to our project.
We follow very simple and clear open-source research community accepted guidelines for contributing.
The guideline instructions divided into sections depending on the part of the project you want to
contribute.

## Rules for adding commits

Create a new branch, if you want to add something new.
The branch naming convention is `<type>/<name of stuff>`, for example
`feat/variance-computation` or `docs/sampling-guide`.
Use one of the types listed below. Additional `/`-separated blocks are allowed, for example
`feat/distributions/registry/add-some-method`. Each block must contain alphanumeric words separated
only by single hyphens (`-`); uppercase letters are allowed. Spaces, underscores, and dots are not
allowed in new branch names.

The `main` branch and branches created by Dependabot and pre-commit.ci are also exempt from
branch naming checks, but their commit messages must follow the same rules.

Commits are added according to conventional commits.
The commit header format is `<type>(<scope>): <description>` or `<type>: <description>`.
The scope is optional; when supplied, it must be nonempty.

The `<type>` field must take one of these values:

* `feat` to add new functionality
* `fix` to fix a bug in the project
* `refactor` for code refactoring, such as renaming a variable
* `test` to add tests, refactor them
* `struct` for changes related to a change in the structure of the project (BUT NOT CODE), for
  example, changing folder locations
* `ci` for various ci/cd tasks
* `docs` for changes in documentation
* `chore` for changes outside the code, for example, gitignore and reamde updates

The `<description>` field contains the gist of the changes in the present imperative in English
without a dot at the end. The first word is a verb starting with a lowercase letter.
Subsequent letters may be capitalized to ensure that abbreviations and terms are displayed correctly.
Keep commit messages concise, informative, and focused on the essence of the change. Prefer a single
line; multiline messages are discouraged but allowed when additional context is necessary.
Automated checks validate the header format, the lowercase first letter of the description, and the
absence of a trailing period. They do not attempt to determine whether the first word is a verb.

Examples:

* Good: "feat: add module for future scrubber implementations"
* Good: "feat(distributions): add variance computation"
* Good: "fix: handle infinite ppf bounds"
* Bad: "Added module for future scrubber implementations."
* Bad: "feat: Add variance computation."

## Source code developers guide (members of PySATL)

1. Install `git` and clone this repo.
2. Build project following build instructions in [README.md](./README.md) file, make sure everything
   is ok.
3. Run tests following instructions in [README.md](./README.md) file, make sure all checks passing.
4. Create new branch from main using branch naming rules.
5. Implement new feature or fix existing one in the source code.
6. Run pre-commit and tests.
7. Commit your changes.
8. Open a pull-request.
9. Wait for review from developers of the project.
10. Fix major and minor issues if presented.
11. Get your work merged into `main`!

## Rules for collaborators

Use the instructions for PySATL members, but use forks instead of branches. You probably won't have
access to create branches in the original repository.

## Basic Tips

1. Don't use merge, only rebase (to keep a linear commit history)
2. Do not change other people's branches unless absolutely necessary
3. Recheck your commit history before creating a pull request
4. Do not use small "fix" commit if you messed up in previous ones. Use `git commit --fixup` and
   `git rebase -i --autosquash` or `git commit --amend` if bad commit is the last one. "fix" commits
   stand for bugs that has been already in master.

### Rules for pull requests

**Forbidden** to merge your pull request into the branch yourself.

Each pull request must be reviewed by one of the maintainers

* Mikhal Mikhailov ([desiment](https://github.com/desiment))
* Vladimir Kutuev ([vkutuev](https://github.com/vkutuev))
* Leonid Elkin ([LeonidElkin](https://github.com/LeonidElkin))

If you click on the green button, then **make sure** that it says `REBASE AND MERGE`!

The review takes place in the form of comments to pull requests, discussions in the team chat and
personal communication. Be polite and don't flood the discussions.
