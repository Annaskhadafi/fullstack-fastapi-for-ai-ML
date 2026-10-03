/**
 * Cloudflare Pages Function: Reverse Proxy to FastAPI Backend
 *
 * Forwards all `/api/*` requests from Cloudflare Pages Edge to the
 * FastAPI backend host (configured in Cloudflare Pages Environment Variable `BACKEND_URL`).
 */
export async function onRequest(context) {
    const { request, env } = context;
    const url = new URL(request.url);

    // Target backend server (default or from Cloudflare environment variable)
    const backendOrigin = env.BACKEND_URL || "https://your-fastapi-backend.vercel.app";
    const targetUrl = new URL(url.pathname + url.search, backendOrigin);

    // Clone headers and preserve client IP / origin
    const newHeaders = new Headers(request.headers);
    newHeaders.set("X-Forwarded-Host", url.host);
    newHeaders.set("X-Forwarded-Proto", url.protocol.replace(":", ""));

    const init = {
        method: request.method,
        headers: newHeaders,
        redirect: "follow",
    };

    if (request.method !== "GET" && request.method !== "HEAD") {
        init.body = request.body;
        // @ts-ignore
        init.duplex = "half";
    }

    try {
        const response = await fetch(targetUrl.toString(), init);
        return new Response(response.body, {
            status: response.status,
            status_code: response.status,
            headers: response.headers
        });
    } catch (err) {
        return new Response(JSON.stringify({
            error: "Cloudflare Pages Gateway Error",
            message: "Gagal menghubungkan ke backend FastAPI: " + err.message
        }), {
            status: 502,
            headers: { "Content-Type": "application/json" }
        });
    }
}
