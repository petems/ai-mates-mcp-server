from __future__ import annotations

from ai_mates_mcp_server.context import read_relevant_files


def test_relevant_file_requires_workspace_root(tmp_path):
    target = tmp_path / "app.py"
    target.write_text("print('ok')", encoding="utf-8")

    result = read_relevant_files([str(target)])

    assert "workspace_root is required" in result
    assert "print('ok')" not in result


def test_reads_relative_file_inside_workspace_root(tmp_path):
    target = tmp_path / "src" / "app.py"
    target.parent.mkdir()
    target.write_text("print('ok')", encoding="utf-8")

    result = read_relevant_files(["src/app.py"], str(tmp_path))

    assert "--- src/app.py ---" in result
    assert "print('ok')" in result


def test_reads_absolute_file_inside_workspace_root(tmp_path):
    target = tmp_path / "app.py"
    target.write_text("print('ok')", encoding="utf-8")

    result = read_relevant_files([str(target)], str(tmp_path))

    assert "--- app.py ---" in result
    assert "print('ok')" in result


def test_blocks_path_traversal_outside_workspace_root(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("top-secret-value", encoding="utf-8")

    result = read_relevant_files(["../secret.txt"], str(workspace))

    assert "blocked: outside workspace root" in result
    assert "top-secret-value" not in result


def test_blocks_sensitive_file_inside_workspace_root(tmp_path):
    target = tmp_path / ".env"
    target.write_text("TOKEN=secret", encoding="utf-8")

    result = read_relevant_files([".env"], str(tmp_path))

    assert "blocked: sensitive file" in result
    assert "TOKEN=secret" not in result


def test_allows_normal_repo_dotfile_inside_workspace_root(tmp_path):
    target = tmp_path / ".gitignore"
    target.write_text(".venv\n", encoding="utf-8")

    result = read_relevant_files([".gitignore"], str(tmp_path))

    assert "--- .gitignore ---" in result
    assert ".venv" in result


def test_directory_expansion_skips_hidden_and_dependency_files(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("app", encoding="utf-8")
    (tmp_path / "src" / "credentials").write_text("top-secret-value", encoding="utf-8")
    (tmp_path / "src" / ".hidden").write_text("hidden", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "package.js").write_text("package", encoding="utf-8")

    result = read_relevant_files(["."], str(tmp_path))

    assert "app" in result
    assert "hidden" not in result
    assert "package" not in result
    assert "top-secret-value" not in result


def test_file_size_cap_blocks_large_file(tmp_path):
    target = tmp_path / "large.txt"
    target.write_text("x" * 1_000_001, encoding="utf-8")

    result = read_relevant_files(["large.txt"], str(tmp_path))

    assert "file too large" in result
    assert "x" * 100 not in result


def test_blocks_nested_sensitive_directory_under_workspace_root(tmp_path):
    project = tmp_path / "project"
    ssh_dir = project / ".ssh"
    ssh_dir.mkdir(parents=True)
    (ssh_dir / "config").write_text("Host secret-host", encoding="utf-8")

    result = read_relevant_files(["project/.ssh/config"], str(tmp_path))

    assert "blocked: sensitive file" in result
    assert "secret-host" not in result


def test_blocks_nested_gcloud_directory_under_workspace_root(tmp_path):
    gcloud_dir = tmp_path / "project" / ".config" / "gcloud"
    gcloud_dir.mkdir(parents=True)
    (gcloud_dir / "credentials.db").write_text("gcloud-token-value", encoding="utf-8")

    result = read_relevant_files(["project/.config/gcloud/credentials.db"], str(tmp_path))

    assert "blocked: sensitive file" in result
    assert "gcloud-token-value" not in result


def test_blocks_sensitive_directory_as_workspace_root(tmp_path):
    sensitive_root = tmp_path / ".aws"
    sensitive_root.mkdir()
    (sensitive_root / "config").write_text("aws-secret-key=top-secret", encoding="utf-8")

    result = read_relevant_files(["config"], str(sensitive_root))

    assert "blocked: invalid workspace root" in result
    assert "sensitive directory" in result
    assert "top-secret" not in result


def test_blocks_gcloud_directory_as_workspace_root(tmp_path):
    gcloud_root = tmp_path / ".config" / "gcloud"
    gcloud_root.mkdir(parents=True)
    (gcloud_root / "access_tokens.db").write_text("gcloud-token-value", encoding="utf-8")

    result = read_relevant_files(["access_tokens.db"], str(gcloud_root))

    assert "blocked: invalid workspace root" in result
    assert "sensitive directory" in result
    assert "gcloud-token-value" not in result


def test_sensitive_patterns_match_case_insensitively(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / ".ENV").write_text("API_KEY=top-secret", encoding="utf-8")
    (project / "SECRET.PEM").write_text("-----BEGIN KEY-----", encoding="utf-8")

    result = read_relevant_files(["project/.ENV", "project/SECRET.PEM"], str(tmp_path))

    assert result.count("blocked: sensitive file") == 2
    assert "top-secret" not in result
    assert "BEGIN KEY" not in result


def test_blocks_directly_requested_excluded_directory_file(tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("url = git@github.com:owner/private.git", encoding="utf-8")

    result = read_relevant_files([".git/config"], str(tmp_path))

    assert "blocked: excluded directory" in result
    assert "owner/private" not in result


def test_blocks_directly_requested_dependency_directory(tmp_path):
    modules = tmp_path / "node_modules" / "pkg"
    modules.mkdir(parents=True)
    (modules / "index.js").write_text("module.exports = 'dependency-body'", encoding="utf-8")

    result = read_relevant_files(["node_modules"], str(tmp_path))

    assert "blocked: excluded directory" in result
    assert "dependency-body" not in result


def test_directory_expansion_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr("ai_mates_mcp_server.context.MAX_WALK_FILES", 3)
    project = tmp_path / "project"
    project.mkdir()
    for index in range(10):
        (project / f"file_{index:02d}.txt").write_text(f"body-{index}", encoding="utf-8")

    result = read_relevant_files(["project"], str(tmp_path))

    assert "directory expansion limit" in result
    assert result.count("body-") == 3
