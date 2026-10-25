# [MODULE DOC]
# Script run by an XLA GitHub workflow after a commit lands. If the head commit message says it
# reverts earlier commits, it finds the pull requests those commits came from and comments on
# each that it was rolled back, then sets its status to open. Uses the sibling github_api
# helper module and reads the access token from the environment. Runs main only when executed
# as a script; no import-time side effects.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Notifies PR authors of rollbacks on committed PRs."""
import os
import re
from typing import Generator, Sequence

import github_api


def get_reverted_commit_hashes(message: str) -> list[str]:
  # [FUNCTION DOC]
  # Extracts commit hashes mentioned after the word "reverts" (case-insensitive, 5 to 40 hex
  # digits) from a commit message. Input: the message text. Returns a list of hash strings
  # (possibly empty). Side effect: prints the message and the found hashes.
  """Searches a commit message for `reverts <commit hash>` and returns the found SHAs.

  Arguments:
    message: the commit message to search

  Returns:
    A list of SHAs as strings.
  """
  print("Head commit message:", message, sep="\n")
  regex = re.compile(r"reverts ([0-9a-f]{5,40})", flags=re.IGNORECASE)
  commit_hashes = regex.findall(message)
  print(f"Found commit hashes reverted in this commit: {commit_hashes}")
  return commit_hashes


def get_associated_prs(
    api: github_api.GitHubAPI,
    commit_hashes: Sequence[str],
) -> Generator[int, None, None]:
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it yields pairs of PR number and SHA as strings, but it
  # actually yields only the PR number as an integer. The match is anchored at the message
  # start, so only messages beginning with the PR marker are found.
  # Inputs: a GitHub API client and a sequence of commit hashes. For each hash it fetches the
  # commit from the openxla/xla repository, looks for a leading PR number in the message and
  # yields that number. Prints progress, and a final message when done (printed even if some
  # were found). Raises HTTP errors from the API client.
  """Finds PRs associated with commits.

  Arguments:
    api: GitHubAPI object which will be used to make requests
    commit_hashes: A sequence of SHAs which may have PRs associated with them

  Yields:
    Associated pairs of (PR number, SHA), both as strings
  """
  regex = re.compile(r"PR #(\d+)")
  for commit_hash in commit_hashes:
    response = api.get_commit("openxla/xla", commit_hash)
    message = response["commit"]["message"]
    if maybe_match := regex.match(message):
      pr_number = maybe_match.group(1)
      print(f"Found PR #{pr_number} associated with commit hash {commit_hash}")
      yield int(pr_number)
  print(f"Didn't find any PRs associated with commit hashes: {commit_hashes}")


def main():
  # [FUNCTION DOC]
  # Workflow entry point; takes no inputs and returns nothing. Builds an API client with the
  # token from the environment, fetches the head commit of openxla/xla, finds reverted commits
  # and their PRs, and for each PR posts a comment saying it was rolled back in the head commit
  # and sets the status to open. Side effects: network calls, comments on GitHub, printing.
  # KNOWN ISSUE: the status update call sends a status field with a POST, which the GitHub API
  # likely ignores, so PRs may not actually be reopened.
  api = github_api.GitHubAPI(os.getenv("GH_TOKEN"))
  head_commit = api.get_commit("openxla/xla", "HEAD")
  commit_hashes = get_reverted_commit_hashes(head_commit["commit"]["message"])

  for pr_number in get_associated_prs(api, commit_hashes):
    sha = head_commit["sha"]
    api.write_issue_comment(
        "openxla/xla", pr_number, f"This PR was rolled back in {sha}!"
    )
    api.set_issue_status("openxla/xla", pr_number, "open")


if __name__ == "__main__":
  main()
