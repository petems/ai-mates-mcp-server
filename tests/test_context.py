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
