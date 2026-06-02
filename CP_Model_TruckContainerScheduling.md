# Mô hình Quy hoạch ràng buộc (CP) — Truck Container Scheduling

## 1. Dữ liệu (tham số)

- $P=\{1,\dots,N\}$: tập điểm; $t_{ij}\ge 0$: thời gian di chuyển từ $i$ đến $j$.
- $\pi\in P$: bãi rơ-mooc (duy nhất); $\gamma^{+}$: thời gian gắn rơ-mooc, $\gamma^{-}$: thời gian tháo/trả rơ-mooc.
- $K=\{1,\dots,m\}$: tập đầu kéo; $o_k\in P$: bãi đỗ (xuất phát & kết thúc) của đầu kéo $k$.
- $R=\{1,\dots,n\}$: tập yêu cầu. Với mỗi $r$:
  - $q_r\in\{1,2\}$: số "slot" container ($20\text{ft}\to1$, $40\text{ft}\to2$);
  - $a_r$: điểm lấy, $b_r$: điểm trả;
  - $\alpha_r\in\{\text{PC},\text{PCT}\}$: hành động lấy (PC = `PICKUP_CONTAINER`, PCT = `PICKUP_CONTAINER_TRAILER`);
  - $\beta_r\in\{\text{DC},\text{DCT}\}$: hành động trả (DC = `DROP_CONTAINER`, DCT = `DROP_CONTAINER_TRAILER`);
  - $\delta^{a}_r,\ \delta^{b}_r$: thời gian tác nghiệp lấy / trả.
- $Q=2$: sức chứa đầu kéo (1 rơ-mooc: tối đa $2\times20\text{ft}$ hoặc $1\times40\text{ft}$).
- $\alpha$: hằng số lớn (trọng số ưu tiên F1).

## 2. Tập đỉnh (node) của đồ thị lộ trình

- $O_k,\,E_k$: đỉnh xuất phát / kết thúc của đầu kéo $k$ (đều ở vị trí $o_k$).
- $p_r$ (lấy, ở $a_r$) và $d_r$ (trả, ở $b_r$) cho mỗi yêu cầu $r$.
- $\mathcal{T}=\{\tau_1,\dots,\tau_L\}$: các đỉnh **dịch vụ rơ-mooc** ở $\pi$ (gắn rơ-mooc rỗng hoặc trả rơ-mooc rỗng), tùy chọn (có thể không dùng). Chọn $L=2n+2m$ là đủ.

Đặt $\mathcal N$ = tập tất cả đỉnh, $\text{loc}(i)$ là vị trí của đỉnh $i$, $\text{serv}(i)$ là thời gian tác nghiệp tại $i$ ($\delta^a_r,\delta^b_r,\gamma^{+},\gamma^{-}$, hoặc $0$ cho $O_k,E_k$).

## 3. Biến quyết định

- $x_{ij}\in\{0,1\}$: $=1$ nếu một đầu kéo đi thẳng từ $i$ đến $j$ ($i,j\in\mathcal N$).
- $y_i\in K$: đầu kéo phục vụ đỉnh $i$.
- $z_i\in\{0,1\}$: $=1$ nếu đỉnh tùy chọn $i\in\mathcal T$ được sử dụng (đỉnh bắt buộc $p_r,d_r,O_k,E_k$ luôn $=1$).
- $T_i\ge 0$: thời điểm **bắt đầu tác nghiệp** tại đỉnh $i$.
- $\ell_i\in\{0,\dots,Q\}$: số slot đang chở **sau khi rời** đỉnh $i$.
- $h_i\in\{0,1\}$: trạng thái có rơ-mooc gắn vào đầu kéo **sau khi rời** đỉnh $i$.

## 4. Ràng buộc

### 4.1 Luồng & cấu trúc lộ trình
$$\sum_{j} x_{ij}=z_i,\qquad \sum_{j} x_{ji}=z_i \quad \forall i\in\mathcal N\setminus\{O_k,E_k\}$$
$$\sum_j x_{O_k j}=1,\qquad \sum_i x_{i E_k}=1 \quad \forall k\in K$$
- Mỗi đầu kéo: một đường đi $O_k\rightsquigarrow E_k$. Cấm chu trình con (ràng buộc circuit / loại trừ subtour, ví dụ `AddCircuit` trong OR-Tools, hoặc `path`/MTZ).
- Lan truyền đầu kéo: $x_{ij}=1\Rightarrow y_i=y_j$;  $y_{O_k}=y_{E_k}=k$.

### 4.2 Gán & thứ tự yêu cầu
$$y_{p_r}=y_{d_r}\quad\forall r \qquad\text{(lấy và trả cùng một đầu kéo)}$$
$$T_{d_r}\ \ge\ T_{p_r}+\delta^{a}_r+t_{a_r b_r}\quad\forall r \qquad\text{(lấy trước trả)}$$

### 4.3 Lan truyền thời gian (theo cung)
$$x_{ij}=1\ \Rightarrow\ T_j\ \ge\ T_i+\text{serv}(i)+t_{\text{loc}(i),\text{loc}(j)}$$
$$T_{O_k}=0\quad\forall k$$

### 4.4 Sức chứa (slot)
$$x_{ij}=1\Rightarrow \ell_j=\ell_i+\Delta_j,\qquad 0\le \ell_i\le Q$$
trong đó $\Delta_{p_r}=+q_r$, $\Delta_{d_r}=-q_r$, $\Delta_i=0$ với các đỉnh khác; $\ell_{O_k}=0$.

### 4.5 Trạng thái rơ-mooc $h$
Khởi tạo $h_{O_k}=0$ (đầu kéo xuất phát không có rơ-mooc). Theo từng loại đỉnh (áp dụng khi $x_{ij}=1$, $h$ trước là $h_i$):

| Đỉnh | Điều kiện trước | $h$ sau |
|------|-----------------|---------|
| Gắn rơ-mooc rỗng $\tau\in\mathcal T$ (gắn) | $h_i=0,\ \ell=0$ | $1$ |
| Trả rơ-mooc rỗng $\tau\in\mathcal T$ (trả) | $h_i=1,\ \ell=0$ | $0$ |
| $p_r$ với $\alpha_r=\text{PC}$ | $h_i=1$ (đã có rơ-mooc rỗng còn chỗ) | $1$ |
| $p_r$ với $\alpha_r=\text{PCT}$ | $h_i=0$ (đầu kéo trần, móc cả rơ-mooc có hàng) | $1$ |
| $d_r$ với $\beta_r=\text{DC}$ | $h_i=1$ | $1$ |
| $d_r$ với $\beta_r=\text{DCT}$ | $h_i=1$ | $0$ (để lại rơ-mooc + container) |
| $E_k$ | $h=0,\ \ell=0$ | — |

> Ghi chú: để phục vụ một yêu cầu $\alpha_r=\text{PC}$, đầu kéo phải ghé $\pi$ gắn rơ-mooc rỗng trước (một đỉnh $\tau$). Trước khi về bãi $E_k$ phải trả hết rơ-mooc ($h=0$). Mỗi đầu kéo chỉ kéo **một** rơ-mooc tại một thời điểm.

## 5. Hàm mục tiêu

$$C_k = T_{E_k}\quad\text{(thời điểm hoàn thành của đầu kéo }k)$$
$$F_1=\max_{k\in K} C_k,\qquad F_2=\sum_{i\in\mathcal N}\sum_{j\in\mathcal N} x_{ij}\,t_{\text{loc}(i),\text{loc}(j)}$$
$$\boxed{\ \min\ F=\alpha\,F_1+F_2\ }$$

(Score = $10^9 - F$.)

## 6. Gợi ý cài đặt
- **OR-Tools CP-SAT**: dùng `AddCircuit` cho mỗi đầu kéo (hoặc đồ thị gộp + biến `y`), `AddElement`/`OnlyEnforceIf` cho lan truyền thời gian, biến nguyên cho $\ell,h$, `AddMaxEquality` cho $F_1$.
- **CP Optimizer**: dùng `intervalVar` cho mỗi tác nghiệp, `sequenceVar`+`noOverlap` với ma trận chuyển tiếp $t_{ij}$ cho mỗi đầu kéo, `endOf` cho $F_1$.
- **MiniZinc**: mảng `succ[]` + `circuit`, cùng các ràng buộc trên.
