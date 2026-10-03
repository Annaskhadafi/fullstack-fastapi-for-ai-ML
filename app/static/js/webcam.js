/**
 * Webcam Controller for Live Computer Vision & HTMX Ingestion
 */
class WebcamController {
    constructor(videoElementId, canvasElementId, hiddenInputId) {
        this.video = document.getElementById(videoElementId);
        this.canvas = document.getElementById(canvasElementId);
        this.hiddenInput = document.getElementById(hiddenInputId);
        this.stream = null;
        this.liveInterval = null;
    }

    async startCamera() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert("Browser Anda tidak mendukung akses kamera atau fitur ini memerlukan HTTPS/localhost.");
            return false;
        }

        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }
            });
            this.video.srcObject = this.stream;
            await this.video.play();
            return true;
        } catch (err) {
            console.error("Camera access error:", err);
            alert("Tidak dapat mengakses kamera: " + err.message);
            return false;
        }
    }

    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        if (this.liveInterval) {
            clearInterval(this.liveInterval);
            this.liveInterval = null;
        }
    }

    captureFrame() {
        if (!this.stream || !this.video) return null;

        const width = (this.video.videoWidth && this.video.videoWidth > 0) ? this.video.videoWidth : 640;
        const height = (this.video.videoHeight && this.video.videoHeight > 0) ? this.video.videoHeight : 480;

        this.canvas.width = width;
        this.canvas.height = height;

        const ctx = this.canvas.getContext("2d");
        ctx.drawImage(this.video, 0, 0, width, height);

        const dataUrl = this.canvas.toDataURL("image/jpeg", 0.85);
        if (this.hiddenInput) {
            this.hiddenInput.value = dataUrl;
        }
        return dataUrl;
    }

    toggleLiveDetection(formId, intervalMs = 1500) {
        if (this.liveInterval) {
            clearInterval(this.liveInterval);
            this.liveInterval = null;
            return false;
        } else {
            this.captureFrame();
            const form = document.getElementById(formId);
            if (form && window.htmx) {
                htmx.trigger(form, 'submit');
            }
            this.liveInterval = setInterval(() => {
                this.captureFrame();
                if (form && window.htmx) {
                    htmx.trigger(form, 'submit');
                }
            }, intervalMs);
            return true;
        }
    }
}

window.WebcamController = WebcamController;
