# Obsidian 特殊语法与代码块总结

Obsidian 不仅仅是一个 Markdown 编辑器，它通过一系列特殊的语法和代码块，极大地增强了笔记的组织能力、可读性和功能性。本指南总结了其中最常用和最强大的部分。

---

## 1. Callouts (标注块)

Callouts 可以在视觉上突出显示特定信息，传达不同的语气和重要性。

### 基础语法

```markdown
> [!TYPE]
> 这是标注块的内容。
```

### 常用类型

| 类型 (Type) | 语法示例 | 外观与用途 |
| :--- | :--- | :--- |
| **Note** | `> [!NOTE]` | **蓝色，铅笔图标**。用于普通的注释或旁注。 |
| **Abstract** | `> [!ABSTRACT]` | **蓝色，剪贴板图标**。用于写摘要、总结或“太长不看”(TLDR)。 |
| **Info** | `> [!INFO]` | **蓝色，信息图标**。用于提供一般信息。 |
| **Todo** | `> [!TODO]` | **蓝色，待办事项图标**。用于标记需要完成的任务。 |
| **Tip** | `> [!TIP]` | **绿色，火焰图标**。用于提供有用的技巧、提示或建议。 |
| **Success** | `> [!SUCCESS]` | **绿色，对勾图标**。用于标记成功完成的事项或正面结果。 |
| **Question** | `> [!QUESTION]` | **黄色，问号图标**。用于提出问题、常见问题解答(FAQ)。 |
| **Warning** | `> [!WARNING]` | **黄色，警告图标**。用于提醒需要注意的事项。 |
| **Failure** | `> [!FAILURE]` | **红色，叉号图标**。用于标记失败的结果、错误。 |
| **Danger** | `> [!DANGER]` | **红色，感叹号图标**。用于非常重要的警告，表示危险。 |
| **Bug** | `> [!BUG]` | **红色，虫子图标**。专门用于记录代码或流程中的 Bug。 |
| **Example** | `> [!EXAMPLE]` | **紫色，列表图标**。用于提供具体的示例或案例。 |
| **Quote** | `> [!QUOTE]` | **灰色，引号图标**。用于引用他人的话或书中的段落。 |

### 高级用法

- **自定义标题**: `> [!TIP] 我的独家秘笈`
- **默认折叠**: `> [!EXAMPLE]- 点击查看代码示例`
- **默认展开**: `> [!NOTE]+ 这是一条默认展开的笔记`

---

## 2. YAML Frontmatter (元数据区)

位于笔记最顶端，由两行 `---` 包裹，用于定义笔记的元数据，供 Obsidian 和插件使用。

### 语法示例

```yaml
---
tags:
  - literature-note
  - 领域/图像分类
aliases: [ViT, Vision Transformer]
cssclass: wide-page
creation_date: 2025-06-21
---
```

- **tags**: 为笔记添加标签，支持层级。
- **aliases**: 为笔记设置别名，可以通过别名链接到此笔记。
- **cssclass**: 应用自定义的 CSS 样式。
- 你也可以添加任何自定义的元数据，供 Dataview 等插件查询。

---

## 3. 任务列表 (Task Lists)

用于创建可交互的待办事项列表。

### 语法

```markdown
- [ ] 未完成的任务
- [x] 已完成的任务
- [-] 已取消的任务
```

---

## 4. Dataview 插件查询 (需要安装 Dataview 插件)

Dataview 可以将你的 Obsidian 仓库变成一个数据库，让你能够动态地查询和展示笔记。

### 语法

````markdown
```dataview
LIST
FROM #literature-note AND #领域/图像分类 
WHERE file.mtime > date(today) - dur(7 days)
SORT file.mtime DESC
```
