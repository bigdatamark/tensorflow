# [MODULE DOC]
# HELPER: GitHub API client used by the XLA GitHub workflow scripts (e.g. rollback notification).
# Provides a small class wrapping the GitHub REST API with only the endpoints needed: get a
# commit, comment on an issue or pull request, set an issue status, add labels, and list a
# user's public organizations. Built on the requests library; no import-time side effects.
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
"""Provides a Python interface to various parts of the GitHub API.

NOTE: not using PyGithub https://github.com/PyGithub/PyGithub as it doesn't
have full API coverage and it's easy enough to use the endpoints we need like
this (with zero dependencies as a bonus!)
"""
from typing import Any, Optional
import urllib

import requests


class GitHubAPI:
  # [CLASS DOC]
  # Thin client for the GitHub REST API. Holds one HTTP session preconfigured with the GitHub
  # JSON accept header and, when a token is supplied, token authorization. Each public method
  # builds an endpoint path and sends it through the private request helper, returning the
  # decoded JSON response. Without a token only unauthenticated (rate limited, read-only)
  # calls will work.
  """Wraps the GitHub REST API."""

  def __init__(self, token: Optional[str] = None):
    # [FUNCTION DOC]
    # Constructor. Input: optional GitHub access token string (default none). Creates an HTTP
    # session, sets the JSON accept header, and sets the authorization header when a token is
    # given. Stores the session on the instance. Returns nothing.
    self._session = requests.Session()
    self._session.headers["Accept"] = "application/vnd.github+json"
    if token:
      self._session.headers["Authorization"] = f"token {token}"

  def _make_request(
      self, verb: str, endpoint: str, **kwargs: dict[str, Any]
  ) -> requests.Response:
    # [FUNCTION DOC]
    # HELPER: shared request sender for all public methods of this class.
    # KNOWN ISSUE: the annotation claims a response object is returned, but the method returns
    # the decoded JSON body (a mapping or list). The keyword argument annotation is also odd.
    # Inputs: HTTP verb, endpoint path relative to the GitHub API root, and keyword arguments
    # that are sent as the JSON request body. Sends the request on the session, raises an HTTP
    # error for failing status codes, and returns the decoded JSON body.
    """Helper method to make a request and raise an HTTPError if one occurred.

    Arguments:
      verb: The HTTP verb to use
      endpoint: The endpoint to make the request to
      **kwargs: The json that will be sent as the body of the request.

    Returns:
      a requests.Response object containing the response from the API.

    Raises:
      requests.exceptions.HTTPError
    """
    res = self._session.request(
        verb,
        urllib.parse.urljoin("https://api.github.com", endpoint),
        json=kwargs,
    )
    res.raise_for_status()
    return res.json()

  def get_commit(self, repo: str, commit_id: str) -> requests.Response:
    # [FUNCTION DOC]
    # KNOWN ISSUE: annotated as returning a response object but returns decoded JSON.
    # Fetches one commit. Inputs: repo as owner/name text and a commit id (hash or ref such as
    # HEAD). Returns the decoded JSON description of the commit. Raises an HTTP error on failure.
    # Note it sends a GET request that also carries an (empty) JSON body.
    """Gets a commit by it's SHA-1 hash.

    https://docs.github.com/en/rest/commits/commits?apiVersion=2022-11-28#get-a-
    commit

    Arguments:
      repo: a string of the form `owner/repo_name`, e.g. openxla/xla.
      commit_id: a string describing the commit to get, e.g. `deadbeef` or
        `HEAD`.

    Returns:
      a requests.Response object containing the response from the API.

    Raises:
      requests.exceptions.HTTPError
    """
    endpoint = f"repos/{repo}/commits/{commit_id}"
    return self._make_request("GET", endpoint)

  def write_issue_comment(
      self, repo: str, issue_number: int, body: str
  ) -> requests.Response:
    # [FUNCTION DOC]
    # KNOWN ISSUE: annotated as returning a response object but returns decoded JSON.
    # Posts a new comment on an issue or pull request. Inputs: repo as owner/name text, issue
    # or PR number, and comment body text. Side effect: creates a comment on GitHub (needs a
    # token with write access). Returns the decoded JSON of the created comment; raises an HTTP
    # error on failure.
    """Writes a comment on an issue (or PR).

    https://docs.github.com/en/rest/issues/comments?apiVersion=2022-11-
    28#create-an-issue-comment

    Arguments:
      repo: a string of the form `owner/repo_name`, e.g. openxla/xla
      issue_number: the issue (or PR) to comment on
      body: the body of the comment

    Returns:
      a requests.Response object containing the response from the API.

    Raises:
      requests.exceptions.HTTPError
    """
    endpoint = f"repos/{repo}/issues/{issue_number}/comments"
    return self._make_request("POST", endpoint, body=body)

  def set_issue_status(
      self, repo: str, issue_number: int, status: str
  ) -> requests.Response:
    # [FUNCTION DOC]
    # KNOWN ISSUE: annotated as returning a response object but returns decoded JSON. Also the
    # GitHub issue update endpoint expects the field named state, not status, and is normally
    # called with PATCH; this sends POST with a status field, so it likely does not change state.
    # Intended to set an issue or PR state. Inputs: repo as owner/name text, issue number and
    # the status text. Returns decoded JSON; raises an HTTP error on failure.
    """Sets the status of an issue (or PR).

    https://docs.github.com/en/rest/issues/issues?apiVersion=2022-11-28#update-
    an-issue

    Arguments:
      repo: a string of the form `owner/repo_name`, e.g. openxla/xla
      issue_number: the issue (or PR) to set the status of
      status: the status to set

    Returns:
      a requests.Response object containing the response from the API.

    Raises:
      requests.exceptions.HTTPError
    """
    endpoint = f"repos/{repo}/issues/{issue_number}"
    return self._make_request("POST", endpoint, status=status)

  def add_issue_labels(
      self, repo: str, issue_number: int, labels: list[str]
  ) -> requests.Response:
    # [FUNCTION DOC]
    # KNOWN ISSUE: annotated as returning a response object but returns decoded JSON.
    # Adds labels to an issue or PR. Inputs: repo as owner/name text, issue number, and a list of
    # label names. Side effect: modifies labels on GitHub (needs write access). Returns the
    # decoded JSON list of labels; raises an HTTP error on failure.
    """Adds labels to an issue (or PR).

    https://docs.github.com/en/actions/managing-issues-and-pull-requests/adding-labels-to-issues

    Arguments:
      repo: a string of the form `owner/repo_name`, e.g. openxla/xla
      issue_number: the issue (or PR) to set the status of
      labels: the labels to add to the issue

    Returns:
      a requests.Response object containing the response from the API.

    Raises:
      requests.exceptions.HTTPError
    """
    endpoint = f"repos/{repo}/issues/{issue_number}/labels"
    return self._make_request("POST", endpoint, labels=labels)

  def get_user_orgs(self, username: str) -> requests.Response:
    # [FUNCTION DOC]
    # KNOWN ISSUE: annotated as returning a response object but returns decoded JSON; the
    # username is also redundantly sent in the request body of a GET.
    # Lists the public organizations a user belongs to. Input: GitHub username text. Returns the
    # decoded JSON list; raises an HTTP error on failure.
    """Gets all public org memberships for a user.

    https://docs.github.com/en/rest/orgs/orgs?apiVersion=2022-11-28#list-organizations-for-a-user

    Arguments:
      username: The user's GitHub username as a string.

    Returns:
      a requests.Response object containing the response from the API.

    Raises:
      requests.exceptions.HTTPError
    """
    endpoint = f"users/{username}/orgs"
    return self._make_request("GET", endpoint, username=username)
