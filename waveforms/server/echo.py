import asyncio

DEFAULT_LISTEN_PORT = 11923
DEFAULT_BROADCAST_PORT = 11924


class EchoServerProtocol:

    def __init__(self, broadcast_port=DEFAULT_BROADCAST_PORT):
        self.broadcast_port = broadcast_port

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        message = data.decode()
        print('Received %r from %s' % (message, addr))
        print('Send %r to %s' % (message, addr))
        self.transport.sendto(data, addr)
        self.broadcast(data)

    def broadcast(self, data):
        self.transport.sendto(data, ('<broadcast>', self.broadcast_port))


async def main(port=DEFAULT_LISTEN_PORT,
               broadcast_port=DEFAULT_BROADCAST_PORT):
    print("Starting UDP server")

    # Get a reference to the event loop as we plan to use
    # low-level APIs.
    loop = asyncio.get_running_loop()

    # One protocol instance will be created to serve all
    # client requests.
    transport, protocol = await loop.create_datagram_endpoint(
        lambda: EchoServerProtocol(broadcast_port),
        local_addr=('0.0.0.0', port),
        allow_broadcast=True)

    try:
        await asyncio.sleep(3600)  # Serve for 1 hour.
    finally:
        transport.close()
