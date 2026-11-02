from pathlib import Path
from typing import Optional

# 模型训练数据里常见的虚拟工作目录前缀，属于"幻觉路径"，需要剥掉
_VIRTUAL_PREFIXES = ("/workspace", "/mnt/data", "/home/user")

# 项目根目录（utils/ 的上一级）
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _flatten_session_nesting(full_path: Path, session_path: Path) -> str:
    """修正 .../session_x/session_x/file 这类重复嵌套。"""
    parts = full_path.parts
    name = session_path.name
    for i in range(len(parts) - 1):
        if parts[i] == name and parts[i + 1] == name:
            return str(session_path / full_path.name)
    return str(full_path)


def resolve_path(filename: str, session_dir: Optional[str] = None) -> str:
    """统一的文件路径解析工具方法。

    核心策略：**先做语义改写，最后统一用 resolve() + is_relative_to() 兜底**。
    绝不依赖"枚举所有特殊情况"，任何最终越界的路径都会被拉回会话目录。

    处理顺序：
    1. 清洗虚拟前缀 (/workspace, /mnt/data, /home/user)
    2. updated/ 目录：按路径分段判断，解析到项目根的 updated/
    3. 绝对路径：在会话内则保留，越界则视为幻觉路径、落回会话目录
    4. 相对路径：去掉 output/ 前缀与重复的 session 名，拼到会话目录
    5. 最终闸门：resolve() 后必须位于会话目录内，否则只取文件名

    场景示例（session_dir = D:/Project/output/session_123）：
    | 输入                              | 结果                                      |
    |-----------------------------------|-------------------------------------------|
    | /workspace/report.md              | session_123/report.md                     |
    | abc/updated/upload/file.pdf       | Project/updated/upload/file.pdf           |
    | /etc/passwd                       | session_123/passwd（不再放行系统文件）      |
    | ../outside.txt                    | session_123/outside.txt（不再逃出会话）     |
    | session_123/report.md             | session_123/report.md（不重复嵌套）         |
    | output/report.md                  | session_123/report.md                     |
    | sub1/sub2/test.md                 | session_123/sub1/sub2/test.md             |
    | a/../b.md                         | session_123/b.md                          |
    | （无 session_dir）                 | 直接 resolve 的绝对路径                     |

    Args:
        filename: 输入的文件名或路径
        session_dir: 会话上下文目录；为空时不做沙箱约束

    Returns:
        str: 解析后的绝对路径
    """
    path_str = (filename or "").replace("\\", "/")

    # 1. 虚拟路径清洗（按段匹配，避免 /workspaceXYZ 被误伤）
    for prefix in _VIRTUAL_PREFIXES:
        if path_str == prefix or path_str.startswith(prefix + "/"):
            path_str = path_str[len(prefix):].lstrip("/")
            break

    # 2. updated/ 特殊处理：用路径分段判断，避免 my_updated/ 误命中
    raw_parts = [p for p in path_str.split("/") if p not in ("", ".")]
    if "updated" in raw_parts:
        idx = raw_parts.index("updated")
        updated_root = _PROJECT_ROOT / "updated"
        rest = raw_parts[idx + 1:]
        target = updated_root.joinpath(*rest).resolve() if rest else updated_root.resolve()
        if target.is_relative_to(updated_root.resolve()):
            return str(target)
        # updated 目录内的路径也越界了，退回会话目录处理
        path_str = target.name

    path = Path(path_str)

    # 没有会话上下文时无法做沙箱约束，保持原语义
    if not session_dir:
        return str(path.resolve())

    session_path = Path(session_dir).resolve()
    session_name = session_path.name

    # 3. 绝对路径
    if path.is_absolute():
        full_path = path.resolve()
        if full_path == session_path or full_path.is_relative_to(session_path):
            return _flatten_session_nesting(full_path, session_path)
        # 不在会话目录内：视为模型幻觉路径，落回会话目录（下游会报"文件不存在"）
        path = Path(full_path.name)

    # 4. 相对路径：去掉开头的 output/ 前缀与重复的 session 名（可能多次重复）
    parts = [p for p in path.parts if p not in ("", ".")]
    while parts and parts[0] == "output":
        parts = parts[1:]
    while parts and parts[0] == session_name:
        parts = parts[1:]

    candidate = (session_path / Path(*parts)).resolve() if parts else session_path

    # 5. 最终闸门：resolve() 展开 .. 与软链接后再校验，越界只取文件名
    if not candidate.is_relative_to(session_path):
        candidate = session_path / path.name

    return str(candidate)
