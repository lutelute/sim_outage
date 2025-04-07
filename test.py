import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random

# --- パラメータ ---
N = 50  # ノード数
relay_ratio = 0.6  # リレーを設置する確率
relay_taus = [0.2, 0.5, 2.0]  # リレーの時定数候補
delay_coeff = 2.0  # 信号伝搬の距離依存係数
time_step = 0.5
max_time = 30
frames = int(max_time / time_step)

# --- ネットワーク構築 ---
# 放射状＋櫛状のようなツリー＋少し枝
G = nx.random_tree(n=N)
additional_edges = int(N * 0.2)  # 少しだけ冗長枝を追加
nodes = list(G.nodes)
while additional_edges > 0:
    u, v = random.sample(nodes, 2)
    if not G.has_edge(u, v) and nx.shortest_path_length(G, u, v) > 2:
        G.add_edge(u, v)
        additional_edges -= 1

# 座標と距離設定
pos = nx.spring_layout(G, seed=42)
for u, v in G.edges:
    dist = np.linalg.norm(pos[u] - pos[v])
    G.edges[u, v]["length"] = dist

# --- リレーの割当 ---
relay_info = {}
for node in G.nodes:
    if random.random() < relay_ratio:
        relay_info[node] = random.choice(relay_taus)

# --- 状態管理 ---
node_status = {n: "normal" for n in G.nodes}  # 状態: normal / fault / tripped
fault_time = {}  # 事故が発生した時刻
trip_time = {}   # 遮断予定時刻（リレーがある場合）
status_log = []

# --- 初期事故ノード ---
start_node = random.choice(list(G.nodes))
node_status[start_node] = "fault"
fault_time[start_node] = 0
if start_node in relay_info:
    trip_time[start_node] = relay_info[start_node]

# --- シミュレーション本体 ---
for t_idx in range(frames):
    current_time = t_idx * time_step
    new_faults = []
    to_trip = []

    # 1. リレー動作（現在時刻が遮断予定時刻を超えたら）
    for node in list(trip_time.keys()):
        if node_status[node] == "fault" and current_time >= trip_time[node]:
            node_status[node] = "tripped"
            del trip_time[node]

    # 2. 隣接ノードが赤なら遮断予定を登録（リレー持ちに限る）
    for node in G.nodes:
        if node_status[node] == "normal":
            for neighbor in G.neighbors(node):
                if node_status[neighbor] == "fault":
                    if node in relay_info and node not in trip_time:
                        trip_time[node] = current_time + relay_info[node]

    # --- 3. 信号波及（赤→赤）---
    for node in G.nodes:
        if node_status[node] != "normal":
            continue
        for neighbor in G.neighbors(node):
            if node_status[neighbor] == "fault":
                delay = G.edges[node, neighbor]["length"] * delay_coeff
                arrival_time = fault_time[neighbor] + delay
                # ⚠️ リレーが間に合っている場合は波及させない
                if node in trip_time and trip_time[node] <= arrival_time:
                    continue  # 遮断済 → 波及ブロック
                if current_time >= arrival_time:
                    new_faults.append((node, current_time))
                    break

    # 4. 状態更新（波及）
    for node, f_time in new_faults:
        node_status[node] = "fault"
        fault_time[node] = f_time
        if node in relay_info:
            trip_time[node] = f_time + relay_info[node]

    # ログ保存
    status_log.append(node_status.copy())

# --- アニメーション描画 ---
fig, ax = plt.subplots(figsize=(8, 8))

def update(frame_idx):
    ax.clear()
    current_state = status_log[frame_idx]
    colors = []
    for node in G.nodes:
        status = current_state[node]
        if status == "normal":
            colors.append("lightgray")
        elif status == "fault":
            colors.append("red")
        elif status == "tripped":
            colors.append("blue")
    ax.set_title(f"Time: {frame_idx * time_step:.1f}")
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3)
    nx.draw_networkx_nodes(G, pos, node_color=colors, ax=ax, node_size=100)
    nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)

ani = animation.FuncAnimation(fig, update, frames=frames, interval=300, repeat=False)
plt.show()
