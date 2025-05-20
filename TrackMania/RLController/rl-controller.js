// <reference path="OpenPlanetNext/Scripts/API/OpenPlanet.js" />
// <reference path="OpenPlanetNext/Scripts/API/Trackmania.js" />

const WebSocket = require('ws') // Usaremos WebSocket para comunicación en tiempo real
let socket

function main() {
  log('Conectando con servidor RL...')
  socket = new WebSocket('ws://localhost:8765') // Mismo puerto que Python

  socket.onopen = () => log('Conexión establecida con Python.')
  socket.onerror = (err) => log('Error en WebSocket: ' + err.message)
  socket.onmessage = (event) => {
    const action = JSON.parse(event.data)
    applyAction(action) // Aplica la acción recibida de Python
  }
}

function applyAction(action) {
  // Control continuo (para PPO/DDPG)
  if (action.steer !== undefined) {
    Player.SetSteer(action.steer) // -1 (izq) a 1 (der)
  }
  if (action.gas !== undefined) {
    Player.SetGas(action.gas) // 0 a 1
  }

  // Opción discreta (para DQN)
  if (action.command === 'left') {
    Player.SetInput(PlayerInput.SteerLeft, true)
  }
  // ... (otros comandos)
}

function update() {
  // Enviar estado del juego a Python cada frame (o cada X ms)
  if (socket && socket.readyState === WebSocket.OPEN) {
    const state = getGameState()
    socket.send(JSON.stringify(state))
  }
}

function getGameState() {
  const vehicle = Player.Vehicle
  return {
    speed: vehicle.Speed,
    position: [vehicle.Position.X, vehicle.Position.Y, vehicle.Position.Z],
    rotation: vehicle.Rotation,
    currentCheckpoint: Race.GetCurrentCheckpoint(),
    raceTime: Race.Time
  }
}

function log(msg) {
  Console.WriteLine('[RL Controller] ' + msg)
}

// Iniciar al cargar el script
main()
