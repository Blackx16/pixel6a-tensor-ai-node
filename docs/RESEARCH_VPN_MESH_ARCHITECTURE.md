# Research Document: Universal Multi-User VPN Mesh Architecture (`v0.3`) for Static GitHub Pages & Android 17 Local AI Hosting

## 1. Architectural Problem Statement

How can a **single public static web app** hosted on GitHub Pages ([`https://blackx16.github.io/pixel6a-tensor-ai-node/`](https://blackx16.github.io/pixel6a-tensor-ai-node/)) allow **any number of independent users** to connect simultaneously to their own phone's AI server over a VPN (such as **Tailscale**, **WireGuard**, **ZeroTier**, or **OpenVPN**) without interfering with one another or routing traffic through a central server?

## 2. Why Static GitHub Pages + Client-Side VPN Routing Is Naturally Multi-Tenant

1. **Browser-Local Execution Model**:
   - GitHub Pages serves static HTML/CSS/JS assets to each visitor's browser. Once loaded, the page executes entirely inside that visitor's local OS networking stack.
   - When **User A** (connected to User A's Tailscale tailnet or WireGuard/OpenVPN tunnel) clicks **Connect**, the browser's `fetch("http://100.x.y.z:8080/v1/...")` call is routed by **User A's local kernel routing table** through User A's virtual `tun0` / `utun` / `wg0` interface directly to User A's Pixel 6a.
   - Simultaneously, when **User B** (connected to User B's separate VPN) opens the exact same [`https://blackx16.github.io/pixel6a-tensor-ai-node/`](https://blackx16.github.io/pixel6a-tensor-ai-node/) URL, User B's browser routes `fetch()` requests through **User B's local VPN interface** to User B's phone.
   - Even if two OpenVPN or WireGuard users configure the **exact same private subnet IP** (e.g., `10.8.0.2:8080`), their traffic is 100% isolated inside their respective encrypted tunnels.
2. **Zero-Leakage URL Hash Pairing (`#vpn=...&key=...`)**:
   - Per RFC 3986, URI fragment identifiers (`#...`) are processed exclusively by the client browser and are **never transmitted in HTTP requests** to `github.io`.
   - In `v0.3`, the Pixel 6a app generates a shareable pairing link using the URL hash (`https://blackx16.github.io/pixel6a-tensor-ai-node/#vpn=100.x.y.z:8080&key=sk-...&proto=tailscale`), and the web app persists multiple named VPN profiles in each user's local `localStorage`.

## 3. Protocol Comparison: Tailscale vs. WireGuard vs. ZeroTier vs. OpenVPN on Android 17

| VPN Protocol / App | Underlying Crypto & Tunnel | Default IP Address Space | Android Interface Name | NAT Traversal & Setup Complexity |
| :--- | :--- | :--- | :--- | :--- |
| **[Tailscale (`com.tailscale.ipn`)](https://tailscale.com/docs/concepts/ip-and-dns-addresses)** | WireGuard (Noise_IK, ChaCha20-Poly1305) | **RFC 6598 CGNAT `100.64.0.0/10`** (`100.64.x.x` – `100.127.x.x`) & IPv6 ULA `fd7a:115c:a1e0::/48` ([Tailscale IP Pool](https://tailscale.com/docs/reference/ip-pool)) | `tun0` / `tailscale0` | **Best Zero-Config Choice**: Automatic STUN/ICE hole-punching + DERP relay fallback across 5G/CGNAT with stable `100.x.y.z` IPs and MagicDNS ([Tailscale CGNAT Docs](https://tailscale.com/docs/reference/cgnat-interoperability)). |
| **[WireGuard (`com.wireguard.android`)](https://developer.android.com/develop/connectivity/vpn)** | Kernel/Userspace WireGuard | Custom RFC 1918 (`10.x.x.x/24`, `172.16.x.x`) | `tun0` / `wg0` | **Lowest Overhead**: Minimal CPU/battery footprint, requires user-managed peer endpoint or hub VPS. |
| **[ZeroTier One (`com.zerotier.one`)](https://forum.mikrotik.com/t/cgnat-ip-range-conflict-between-starlink-and-tailscale-site-to-site-vpn/174642)** | VL1 / VL2 Layer-2 Ethernet Virtualization | Managed RFC 1918 (`10.147.x.x`, `172.25.x.x`) or `100.64.0.0/10` | `tun0` / `zt*` | **Best Layer-2 Mesh**: Supports broadcast/multicast and custom subnet assignment across peers. |
| **[OpenVPN (`net.openvpn.openvpn`)](https://developer.android.com/develop/connectivity/vpn)** | TLS 1.3 + AES-256-GCM over UDP/TCP | Custom RFC 1918 (`10.8.0.0/24`) | `tun0` / `tap0` | **Best Legacy Firewall Compatibility**: Can tunnel over TCP port 443 when UDP is blocked by restrictive corporate/campus networks. |

## 4. Solving the 3 Critical Android & Browser Networking Obstacles in `v0.3`

### 4.1 Obstacle 1: Why Standard Java `isSiteLocalAddress()` Rejects Tailscale (`100.64.0.0/10`)
* **The Bug in Standard Android Apps**: Java's `Inet4Address.isSiteLocalAddress()` only checks RFC 1918 ranges (`10/8`, `172.16/12`, `192.168/16`). Because Tailscale assigns addresses from the **RFC 6598 Shared CGNAT range (`100.64.0.0/10`)** ([Tailscale Reserved IPs](https://tailscale.com/docs/reference/reserved-ip-addresses), [WinDock CGNAT Issue #23](https://github.com/aliihsaad/WinDock/issues/23)), `isSiteLocalAddress()` returns `false` and ignores the Tailscale IP!
* **Our Fix in `TensorHardwareOptimizer.kt` (`v0.3`)**:
  - Added `isCgnatOrTailscaleIpv4(addr: Inet4Address)` which inspects raw octets: `firstOctet == 100 && (secondOctet in 64..127)`.
  - Added `scanAllNetworkEndpoints()` using both `ConnectivityManager.getAllNetworks()` (`NetworkCapabilities.TRANSPORT_VPN`) ([Android VPN Guide](https://developer.android.com/develop/connectivity/vpn), [Talsec Kotlin VPN Detection](https://docs.talsec.app/appsec-articles/articles/how-to-detect-vpn-using-kotlin)) and `NetworkInterface.getNetworkInterfaces()` (`tun*`, `wg*`, `tailscale*`, `zt*`, `tap*`, `ppp*`, `wlan*`) ([SkillsDirectory VPN Detector](https://www.skillsdirectory.com/skills/reason-machines-vpn-detector-android)).
  - Why we do **not** declare our own `VpnService`: Android allows only one active `VpnService` per user profile ([Android VPN Guide](https://developer.android.com/develop/connectivity/vpn)). By binding `ServerSocket("0.0.0.0", 8080)` as a standard socket server while Tailscale, WireGuard, ZeroTier, or OpenVPN owns `tun0`, the user's VPN stays connected and routes incoming VPN packets directly into `OpenAiHttpServer`.

### 4.2 Obstacle 2: W3C Private Network Access (PNA) & Local Network Access (LNA) Preflights
* **Browser Rule**: When a public `https://` origin (`https://blackx16.github.io`) sends a `fetch()` request to a private or CGNAT IP (`100.64.0.0/10`, `10.0.0.0/8`, `192.168.0.0/16`), Chromium browsers send a preflight `OPTIONS` request containing `Access-Control-Request-Private-Network: true` ([W3C Private Network Access Spec](https://wicg.github.io/private-network-access/), [Chrome PNA Preflight Blog](https://developer.chrome.com/blog/private-network-access-preflight), [W3C Local Network Access Spec](https://wicg.github.io/local-network-access/)).
* **Our Fix in `OpenAiHttpServer.kt` (`v0.3`)**: Every HTTP response and `OPTIONS` preflight includes `Access-Control-Allow-Private-Network: true`, `Access-Control-Allow-Origin: *`, `Access-Control-Allow-Methods: GET, POST, OPTIONS`, and `Access-Control-Allow-Headers: Authorization, Content-Type, X-Tensor-VPN-Mode`.

### 4.3 Obstacle 3: HTTPS-to-HTTP Mixed Content Bypass via Full In-Tunnel Web Console (`http://<vpn-ip>:8080/`) & Tailscale HTTPS
* **Browser Rule**: Strict browsers block `https://blackx16.github.io` from fetching plain `http://100.x.y.z:8080` unless either:
  1. The user enables **Insecure content = Allow** for `blackx16.github.io`,
  2. The user exposes the port via **Tailscale Serve (`https://<node>.<tailnet>.ts.net`)** ([Tailscale Serve CLI](https://tailscale.com/docs/reference/tailscale-cli/serve), [SSD Nodes Serve vs Funnel](https://www.ssdnodes.com/learn/tailscale-serve-vs-funnel)), OR
  3. The user opens the Web Console **directly over the encrypted VPN tunnel** at `http://<vpn-ip>:8080/`!
* **Our Fix in `OpenAiHttpServer.kt` (`v0.3`)**:
  - We bundle the entire interactive Web Console inside the Android APK at `assets/web/index.html`.
  - When any browser on the VPN visits `http://<vpn-ip>:8080/` (or clicks **Direct HTTP ↗** on the GitHub Pages site), `OpenAiHttpServer` serves the complete interactive web app directly from the Pixel 6a over the encrypted WireGuard/OpenVPN `tun0` interface—auto-detecting `window.location.host` (`<vpn-ip>:8080`) with zero Mixed-Content or CORS restrictions!
