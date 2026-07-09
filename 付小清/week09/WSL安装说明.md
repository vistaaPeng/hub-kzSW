# WSL2 + Ubuntu 手动安装说明

## 当前进度（重启后）

- [x] WSL 功能已启用
- [x] 虚拟机平台已启用
- [x] WSL2 内核已更新（`wsl --update`）
- [ ] Ubuntu 22.04 发行版（需手动完成）

## 方法一：Microsoft Store（推荐）

1. 打开 **Microsoft Store**
2. 搜索 **Ubuntu 22.04 LTS**
3. 点击「获取 / 安装」
4. 安装完成后，从开始菜单打开 **Ubuntu 22.04**
5. 首次启动会要求创建 Linux 用户名和密码（记住密码）

## 方法二：命令行（需网络可访问 GitHub）

```powershell
wsl --install -d Ubuntu-22.04
```

若报 `WININET_E_NAME_NOT_RESOLVED`，说明无法访问 GitHub，请用方法一。

## Ubuntu 安装完成后

在 Ubuntu 终端执行：

```bash
bash "/mnt/e/DeepLearning/week9/week9 大模型应用补充知识/work9/setup_wsl_inside.sh"
```

## 验证 WSL + GPU

```bash
nvidia-smi
# 应显示 GTX 1060
```

## 重要：本机 GPU 限制

GTX 1060（Pascal, sm_61）**不满足 vLLM 官方最低要求**（需 sm_70+，Volta 及以上）。

因此：
- **本机可完成**：transformers baseline 实测（已在 `outputs/` 中）
- **vLLM 完整部署**：需换 RTX 20 系列及以上 GPU，或使用云服务器

作业提交可直接使用 `work9/` 中已有结果 + 本说明中的部署脚本。
