const express = require('express');
const WebSocket = require('ws');
const path = require('path');
const { spawn } = require('child_process');

const app = express();
const PORT = 3000;
const WS_PORT = 8080;

// Serve static files
app.use(express.static('public'));

// HTTP Server
const server = app.listen(PORT, () => {
    console.log(`🌐 UI Server: http://localhost:${PORT}`);
});

// WebSocket Server
const wss = new WebSocket.Server({ port: WS_PORT });
console.log(`🔌 WebSocket: ws://localhost:${WS_PORT}`);

let pythonProcess = null;
let clients = new Set();

wss.on('connection', (ws) => {
    console.log('👤 Client connected');
    clients.add(ws);
    
    ws.send(JSON.stringify({ 
        type: 'status', 
        status: 'connected', 
        message: 'Connected to server' 
    }));
    
    // Start Python if not running
    if (!pythonProcess) {
        startPython();
    }
    
    ws.on('message', (message) => {
        try {
            const data = JSON.parse(message);
            handleClientMessage(ws, data);
        } catch (e) {
            console.error('Error parsing message:', e);
        }
    });
    
    ws.on('close', () => {
        console.log('👤 Client disconnected');
        clients.delete(ws);
        if (clients.size === 0 && pythonProcess) {
            pythonProcess.kill();
            pythonProcess = null;
            console.log('🐍 Python process stopped');
        }
    });
});

function startPython() {
    console.log('🐍 Starting Python detector...');
    pythonProcess = spawn('python', ['detector_ws.py']);
    
    pythonProcess.stdout.on('data', (data) => {
        const output = data.toString();
        try {
            const jsonData = JSON.parse(output);
            broadcast(jsonData);
        } catch (e) {
            // Not JSON - send as log
            broadcast({ 
                type: 'log', 
                level: 'info', 
                message: output.trim() 
            });
        }
    });
    
    pythonProcess.stderr.on('data', (data) => {
        console.error('Python stderr:', data.toString());
        broadcast({ 
            type: 'log', 
            level: 'error', 
            message: data.toString().trim() 
        });
    });
    
    pythonProcess.on('close', (code) => {
        console.log(`🐍 Python process exited with code ${code}`);
        pythonProcess = null;
    });
}

function handleClientMessage(ws, data) {
    if (pythonProcess && pythonProcess.stdin) {
        pythonProcess.stdin.write(JSON.stringify(data) + '\n');
    }
}

function broadcast(data) {
    const message = JSON.stringify(data);
    clients.forEach(client => {
        if (client.readyState === WebSocket.OPEN) {
            client.send(message);
        }
    });
}

console.log('✅ Server ready!');