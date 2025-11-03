import os
import json
import urllib.parse

# -------------------------
# 配置
# -------------------------
ROOT_DIR = "D:\\tmp\\electronicFile"  # 卷宗根目录
OUTPUT_DIR = "./web"  # 输出 HTML/JSON 目录
PDFJS_PATH = "/pdfjs/web/viewer.html"  # PDF.js viewer 路径


# -------------------------
# 扫描目录并生成树
# -------------------------
def scan_case_dir(root_dir):
    def scan_dir(path):
        items = []
        for name in sorted(os.listdir(path)):
            full_path = os.path.join(path, name)
            if os.path.isdir(full_path):
                items.append({
                    "name": name,
                    "type": "dir",
                    "children": scan_dir(full_path)
                })
            elif name.lower().endswith(".pdf"):
                rel_path = os.path.relpath(full_path, root_dir).replace("\\", "/")
                items.append({
                    "name": name,
                    "type": "file",
                    "url": f"/electronic/{urllib.parse.quote(rel_path)}"
                })
        return items

    catalog = {}
    for case_name in sorted(os.listdir(root_dir)):
        case_path = os.path.join(root_dir, case_name)
        if os.path.isdir(case_path):
            catalog[case_name] = scan_dir(case_path)
    return catalog


# -------------------------
# 保存 JSON
# -------------------------
def save_json(catalog, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "catalog.json"), "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    print("catalog.json 已生成")


# -------------------------
# 保存 HTML
# -------------------------
def save_html(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width,initial-scale=1" />
        <title>电子卷宗目录</title>
        <style>
            body {
                margin: 0;
                font-family: sans-serif;
                background: #121a2e;
                color: #e6e8ee;
            }
    
            .app {
                display: grid;
                grid-template-columns: 360px 1fr;
                height: 100vh;
            }
    
            .sidebar {
                border-right: 1px solid #1f2a44;
                background: #0e1628;
                display: flex;
                flex-direction: column;
            }
    
            .header {
                padding: 14px 16px;
                border-bottom: 1px solid #1f2a44;
            }
    
            .title {
                font-size: 16px;
                font-weight: 700;
            }
    
            .tree-wrap {
                overflow: auto;
                flex: 1;
                padding: 8px;
            }
    
            .tree {
                list-style: none;
                margin: 0;
                padding: 0;
            }
    
            .tree li .row {
                display: flex;
                align-items: center;
                padding: 4px 8px;
                cursor: pointer;
            }
    
            .tree li .row:hover {
                background: rgba(255, 255, 255, 0.04);
            }
    
            .name a {
                color: #e6e8ee;
                text-decoration: none;
            }
    
            .name a:hover {
                color: #4ea1ff;
                text-decoration: underline;
            }
    
            .children {
                list-style: none;
                margin: 0 0 0 16px;
                padding: 0;
                display: none;
            }
    
            .dir.open>.children {
                display: block;
            }
    
            .twisty {
                width: 12px;
                display: inline-block;
                transition: transform 0.2s;
            }
    
            .dir.open>.row .twisty {
                transform: rotate(90deg);
            }
    
            .main {
                display: flex;
                flex-direction: column;
                height: 100vh;
            }
    
            .preview-head {
                padding: 10px 12px;
                border-bottom: 1px solid #1f2a44;
                flex: 0 0 auto;
            }
    
            #current {
                color: #9aa3b2;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
    
            .preview {
                flex: 1 1 auto;
                min-height: 0;
            }
    
            #viewer {
                width: 100%;
                height: 100%;
                border: 0;
                background: #111;
            }
    
            .highlight {
                background: rgba(78, 161, 255, 0.25);
                border-radius: 4px;
            }
        </style>
    </head>
    
    <body>
        <div class="app">
            <aside class="sidebar">
                <div class="header">
                    <div class="title">电子卷宗目录</div>
                </div>
                <div class="tree-wrap">
                    <ul class='tree root'></ul>
                </div>
            </aside>
            <main class="main">
                <div class="preview-head">
                    <div id="current">未选择文件</div>
                </div>
                <div class="preview"><iframe id="viewer" title="PDF 预览"></iframe></div>
            </main>
        </div>
        <script>
            const treeRoot = document.querySelector('.tree.root');
            const viewer = document.getElementById('viewer');
            const current = document.getElementById('current');
    
            // 获取 URL 参数
            const urlParams = new URLSearchParams(window.location.search);
            const caseParam = urlParams.get('case'); // 例：(2026) 鲁民初4号
    
            // 加载目录
            fetch('catalog.json').then(res => res.json()).then(data => {
                treeRoot.innerHTML = '';
    
                if (caseParam && data[caseParam]) {
                    // 只显示指定案件
                    renderCase(caseParam, data[caseParam]);
                } else {
                    // 如果没传参数，就显示全部
                    Object.keys(data).forEach(caseName => {
                        renderCase(caseName, data[caseName]);
                    });
                }
            });
    
            function renderCase(caseName, caseData) {
                const caseLi = document.createElement('li');
                caseLi.className = 'dir open';
                const row = document.createElement('div');
                row.className = 'row';
                row.innerHTML = '<span class="twisty">▶</span><span class="name">' + caseName + '</span>';
                const ul = document.createElement('ul');
                ul.className = 'children';
                caseLi.appendChild(row);
                caseLi.appendChild(ul);
                buildTree(caseData, ul);
                treeRoot.appendChild(caseLi);
    
                // 如果是指定案件，展开第一个文件
                if (caseParam && caseName === caseParam) {
                    const firstFileLink = ul.querySelector('a.file-link');
                    if (firstFileLink) {
                        firstFileLink.click();
                    }
                }
    
                // 点击案件标题展开/收起
                row.addEventListener('click', () => {
                    caseLi.classList.toggle('open');
                });
            }
    
            // 递归构建树
            function buildTree(items, parent) {
                items.forEach(item => {
                    const li = document.createElement('li');
                    li.className = item.type;
                    const row = document.createElement('div');
                    row.className = 'row';
                    if (item.type === 'dir') {
                        row.innerHTML = '<span class="twisty">▶</span><span class="name">' + item.name + '</span>';
                        const ul = document.createElement('ul');
                        ul.className = 'children';
                        li.appendChild(row);
                        li.appendChild(ul);
                        buildTree(item.children, ul);
                        row.addEventListener('click', () => { li.classList.toggle('open'); });
                    } else {
                        row.innerHTML = '<span class="name"><a href="#" class="file-link">' + item.name + '</a></span>';
                        row.querySelector('a').addEventListener('click', e => {
                            e.preventDefault();
                            viewer.src = '/pdfjs/web/viewer.html?file=' + encodeURIComponent(item.url);
                            current.textContent = item.name;
                        });
                        li.appendChild(row);
                    }
                    parent.appendChild(li);
                });
            }
        </script>
    </body>
    
    </html>
    """
    with open(os.path.join(output_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_content)
    print("index.html 已生成")


# -------------------------
# 主程序
# -------------------------
if __name__ == "__main__":
    catalog = scan_case_dir(ROOT_DIR)
    save_json(catalog, OUTPUT_DIR)
    save_html(OUTPUT_DIR)
