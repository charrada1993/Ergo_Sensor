from flask_socketio import SocketIO, emit

socketio = SocketIO(cors_allowed_origins="*")

def register_socket_events(socketio):
    @socketio.on('connect')
    def handle_connect():
        print('Client connected')
        emit('connected', {'data': 'Connected to server'})

    @socketio.on('disconnect')
    def handle_disconnect():
        print('Client disconnected')