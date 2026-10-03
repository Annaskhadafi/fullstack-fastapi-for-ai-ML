/**
 * Antarmuka Utama & Utilitas HTMX
 */
document.addEventListener("DOMContentLoaded", () => {
    // Auto-scroll chat to bottom when updated
    const chatContainer = document.getElementById("chat-messages-container");
    if (chatContainer) {
        const observer = new MutationObserver(() => {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        });
        observer.observe(chatContainer, { childList: true, subtree: true });
    }
});

function copyToClipboard(text, elementId = null) {
    navigator.clipboard.writeText(text).then(() => {
        if (elementId) {
            const el = document.getElementById(elementId);
            if (el) {
                const originalText = el.innerText;
                el.innerText = "Tersalin!";
                setTimeout(() => {
                    el.innerText = originalText;
                }, 2000);
            }
        }
    });
}
