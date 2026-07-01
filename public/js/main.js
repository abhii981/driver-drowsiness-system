class DrowsinessApp {
    constructor() {
        // WebSocket
        this.ws = null;
        this.isConnected = false;
        this.isRunning = false;
        
        // DOM elements
        this.videoFeed = document.getElementById('videoFeed');
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.resetBtn = document.getElementById('resetBtn');
        this.statusText = document.getElementById('statusText');
        this.statusDot = document.getElementById('statusDot');
        this.alertOverlay = document.getElementById('alertOverlay');
        this.logContainer = document.getElementById('logContainer');
        
        // Metrics
        this.earValue = document.getElementById('earValue');
        this.earBar = document.getElementById('earBar');
        this.earStatus = document.getElementById('earStatus');
        this.perclosValue = document.getElementById('perclosValue');
        this.perclosBar = document.getElementById('perclosBar');
        this.perclosStatus = document.getElementById('perclosStatus');
        this.scoreValue = document.getElementById('scoreValue');
        this.scoreBar = document.getElementById('scoreBar');
        this.scoreStatus = document.getElementById('scoreStatus');
        this.eyeStateValue = document.getElementById('eyeStateValue');
        this.eyeStateStatus = document.getElementById('eyeStateStatus');
        
        // Indicators
        this.yawnStatus = document.getElementById('yawnStatus');
        this.nodStatus = document.getElementById('nodStatus');
        this.blinkRate = document.getElementById('blinkRate');
        this.dlStatus = document.getElementById('dlStatus');
        
        // Settings
        this.earThreshold = document.getElementById('earThreshold');
        this.scoreThreshold = document.getElementById('scoreThreshold');
        this.earThresholdLabel = document.getElementById('earThresholdLabel');
        this.scoreThresholdLabel = document.getElementById('scoreThresholdLabel');
        
        // Audio
        this.alarmAudio = new Audio('/audio/alarm.mp3');
        this.alarmAudio.loop = true;
        
        this.init();
    }
    
    init() {
        this.startBtn.addEventListener('click', () => this.startDetection());
        this.stopBtn.addEventListener('click', () => this.stopDetection());
        this.resetBtn.addEventListener('click', () => this.resetSystem());
        
        this.earThreshold.addEventListener('input', (e) => {
            this.earThresholdLabel.textContent = parseFloat(e.target.value).toFixed(2);
            this.sendSettings();
        });
        
        this.scoreThreshold.addEventListener('input', (e) => {
            this.scoreThresholdLabel.textContent = parseFloat(e.target.value).toFixed(2);
            this.sendSettings();
        });
        
        this.connectWebSocket();
        this.addLog('info', '🟢 UI initialized. Connecting to server...');
    }
    
    connectWebSocket() {
        const wsUrl = `ws://${window.location.hostname}:8080`;
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            this.isConnected = true;
            this.updateStatus('Connected', 'connected');
            this.addLog('success', '✅ Connected to server');
            this.startBtn.disabled = false;
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };
        
        this.ws.onclose = () => {
            this.isConnected = false;
            this.updateStatus('Disconnected', '');
            this.addLog('error', '❌ Disconnected from server');
            this.startBtn.disabled = true;
            this.stopBtn.disabled = true;
            setTimeout(() => this.connectWebSocket(), 3000);
        };
    }
    
    handleMessage(data) {
        switch(data.type) {
            case 'video_frame':
                this.videoFeed.src = `data:image/jpeg;base64,${data.frame}`;
                break;
            case 'metrics':
                this.updateMetrics(data.metrics);
                break;
            case 'alert':
                this.handleAlert(data.status);
                break;
            case 'log':
                this.addLog(data.level, data.message);
                break;
            case 'status':
                this.updateStatus(data.message, data.status);
                break;
        }
    }
    
    updateMetrics(metrics) {
        // Update EAR
        this.earValue.textContent = metrics.ear.toFixed(3);
        this.earBar.style.width = `${Math.min(metrics.ear / 0.5 * 100, 100)}%`;
        if (metrics.ear < 0.2) {
            this.earStatus.textContent = '⚠️ Closed';
            this.earStatus.style.color = '#ff1744';
        } else {
            this.earStatus.textContent = '✅ Open';
            this.earStatus.style.color = '#00e676';
        }
        
        // Update PERCLOS
        this.perclosValue.textContent = `${(metrics.perclos * 100).toFixed(1)}%`;
        this.perclosBar.style.width = `${metrics.perclos * 100}%`;
        if (metrics.perclos > 0.3) {
            this.perclosStatus.textContent = '⚠️ High';
            this.perclosStatus.style.color = '#ff1744';
        } else {
            this.perclosStatus.textContent = '✅ Normal';
            this.perclosStatus.style.color = '#00e676';
        }
        
        // Update Score
        this.scoreValue.textContent = `${(metrics.score * 100).toFixed(1)}%`;
        this.scoreBar.style.width = `${metrics.score * 100}%`;
        if (metrics.score > 0.45) {
            this.scoreStatus.textContent = '⚠️ Drowsy';
            this.scoreStatus.style.color = '#ff1744';
        } else {
            this.scoreStatus.textContent = '✅ Safe';
            this.scoreStatus.style.color = '#00e676';
        }
        
        // Update Eye State
        this.eyeStateValue.textContent = metrics.is_closed ? '😴 Closed' : '👁️ Open';
        this.eyeStateValue.style.color = metrics.is_closed ? '#ff1744' : '#00e676';
        this.eyeStateStatus.textContent = metrics.is_closed ? '⚠️ Drowsy' : '✅ Alert';
        this.eyeStateStatus.style.color = metrics.is_closed ? '#ff1744' : '#00e676';
        
        // Update indicators
        this.yawnStatus.textContent = metrics.is_yawn ? '⚠️ Yes' : '✅ No';
        this.yawnStatus.style.color = metrics.is_yawn ? '#ffab00' : '#00e676';
        
        this.nodStatus.textContent = metrics.head_nod ? '⚠️ Yes' : '✅ No';
        this.nodStatus.style.color = metrics.head_nod ? '#ffab00' : '#00e676';
        
        this.blinkRate.textContent = `${metrics.blink_rate || 0}/min`;
        this.dlStatus.textContent = metrics.use_dl ? '✅ ON' : '⏹️ OFF';
        this.dlStatus.style.color = metrics.use_dl ? '#00e676' : '#ffab00';
    }
    
    handleAlert(status) {
        if (status) {
            this.alertOverlay.classList.add('active');
            this.statusDot.className = 'dot drowsy';
            this.updateStatus('⚠️ DROWSY', 'drowsy');
            this.addLog('error', '🚨 DROWSINESS ALERT! Please pull over!');
            this.alarmAudio.play().catch(e => console.log('Audio error:', e));
        } else {
            this.alertOverlay.classList.remove('active');
            this.statusDot.className = 'dot connected';
            this.updateStatus('Alert', 'connected');
            this.alarmAudio.pause();
            this.alarmAudio.currentTime = 0;
        }
    }
    
    updateStatus(text, className) {
        this.statusText.textContent = text;
        if (className) {
            this.statusDot.className = `dot ${className}`;
        }
    }
    
    addLog(level, message) {
        const log = document.createElement('div');
        log.className = `log ${level}`;
        log.textContent = message;
        this.logContainer.appendChild(log);
        
        while (this.logContainer.children.length > 50) {
            this.logContainer.removeChild(this.logContainer.firstChild);
        }
        
        this.logContainer.scrollTop = this.logContainer.scrollHeight;
    }
    
    startDetection() {
        if (!this.isConnected) {
            this.addLog('warning', '⚠️ Not connected to server');
            return;
        }
        
        this.isRunning = true;
        this.startBtn.disabled = true;
        this.stopBtn.disabled = false;
        
        this.ws.send(JSON.stringify({ type: 'start_detection' }));
        this.addLog('info', '🟢 Detection started');
    }
    
    stopDetection() {
        this.isRunning = false;
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
    
        this.ws.send(JSON.stringify({ type: 'stop_detection' }));
        this.addLog('info', '🔴 Detection stopped');
    
    // STOP THE ALARM SOUND
        this.alarmAudio.pause();
        this.alarmAudio.currentTime = 0;
        this.alertOverlay.classList.remove('active');
    }
    
    resetSystem() {
        this.stopDetection();
        this.alertOverlay.classList.remove('active');
        this.updateStatus('Connected', 'connected');
        this.addLog('info', '⟳ System reset');
    }
    
    sendSettings() {
        if (!this.isConnected) return;
        
        this.ws.send(JSON.stringify({
            type: 'update_settings',
            settings: {
                ear_threshold: parseFloat(this.earThreshold.value),
                score_threshold: parseFloat(this.scoreThreshold.value)
            }
        }));
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    new DrowsinessApp();
});