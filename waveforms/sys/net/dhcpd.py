#
# Packet format
# -------------
#
# Offset    Length   Notes
# ------    ------   -----
#
#   0       1        Operation code with 1 being request and 2 being response
#   1       1        Hardware type with 1 being "Ethernet 10Mb"
#   2       1        Hardware address length with 6 for Ethernet
#   3       1        Hops - usually 0 unless DHCP relaying in operation
#   4-7     4        Transaction ID (selected randomly by client)
#   8-9     2        Seconds - might be used by a server to prioritise requests
#  10-11    2        Flags (only most significan bit used for broadcast)
#  12-15    4        Client Internet address (might be requested by client)
#  16-19    4        Your Internet address (the IP assigned by the server)
#  20-23    4        Server Internet address (the IP of the server)
#  24-27    4        Gateway Internet address (if DHCP relaying in operation)
#  28-43    16       Client hardware address - only first 6 bytes used for Ethernet
#  44-107   64       Text name of server (optional)
# 108-235   128      Boot file name (optional - used for PXE booting)
# 236-239   4        Magic cookie (decimal values 99, 130, 83, 99 )
#

#
# DHCP option codes
# -----------------
#
# 1   - Subnet mask
# 3   - Router(s)
# 6   - DNS name server(s)
# 12  - Hostname
# 15  - Domain name
# 28  - Broadcast address
# 33  - Static route
# 42  - Network time protocol servers
# 51  -
# 53  - DHCP Message Type (DHCPDISCOVER, DHCPOFFER, etc)
# 54  - Server identifier
# 55  - Parameter Request List
# 57  - Maximum DHCP Message Size
# 58  - Renewal (T1) time value
# 59  - Renewal (T2) time value
# 60  - Vendor Class Identifier
# 61  - Client Identifier
# 67  - Boot file name (e.g. PXE booting)
# 80  -
# 116 -
# 119 -
# 145 -
# 167 -
# 171 -
#

import socket
import logging

logger = logging.getLogger(__name__)

# constants

OPT_DHCP_MESSAGE_TYPE = 53
MAGIC_COOKIE = b'c\x82Sc'


def showpacket(bytes):
    bpr = 16  # bpr is Bytes Per Row
    numbytes = len(bytes)

    if numbytes == 0:
        print("<empty packet>")
    else:
        i = 0
        while i < numbytes:
            if (i % bpr) == 0:
                print("{:04d} :".format(i), sep='', end='')

            print(" {:02X}".format(bytes[i]), sep='', end='')

            if ((i + 1) % bpr) == 0:
                print()

            i = i + 1

    if (numbytes % bpr) != 0:
        print()


def readablebytes(bytes):
    numbytes = len(bytes)

    if numbytes == 0:
        readable = "0xNull"
    else:
        readable = "0x"
        i = 0
        while i < numbytes:
            readable += "{:02X}".format(bytes[i])
            i += 1

    return readable


def readablemacaddress(bytes):
    numbytes = len(bytes)

    if numbytes != 6:
        readable = "<invalid MAC>"
    else:
        readable = ""
        i = 0
        while i < 6:
            if i > 0:
                readable += ":"
            readable += "{:02X}".format(bytes[i])
            i += 1

    return readable


def showoptions(bytes):
    numbytes = len(bytes)

    i = 0
    while i < numbytes:
        option = bytes[i]
        if option == 0:
            print("PAD")
            i += 1
            continue

        if option == 255:
            print("END")
            break

        i += 1
        if i >= numbytes:
            print("Option:", option,
                  "- premature EOF when expecting length byte")
            break

        optlen = bytes[i]
        i += 1

        if (i + optlen) >= numbytes:
            print("Option:", option, "with length", optlen,
                  "- premature EOF when expecting option data")
            break

        optdata = bytes[i:i + optlen]
        print("Option:", option, "Length:", optlen, "Value:",
              readablebytes(optdata))

        i += optlen


##############################################################################


def buildbyteoption(optnum, ba):
    lenba = len(ba)

    if (ba) == 0:
        opt = bytearray(1)
        opt[0] = optnum
    else:
        opt = bytearray(2 + lenba)
        opt[0] = optnum
        opt[1] = lenba
        opt[2:2 + lenba] = ba

    return opt


def build1byteoption(optnum, databyte):
    optbytes = bytearray(3)
    optbytes[0] = optnum
    optbytes[1] = 1
    optbytes[2] = databyte

    return optbytes


def build4byteoption(optnum, d1, d2, d3, d4):
    optbytes = bytearray(6)
    optbytes[0] = optnum
    optbytes[1] = 4
    optbytes[2] = d1
    optbytes[3] = d2
    optbytes[4] = d3
    optbytes[5] = d4

    return optbytes


def buildstringoption(optnum, string):
    optbytes = bytearray(2 + len(string))
    optbytes[0] = optnum
    optbytes[1] = len(string)
    d = 2
    for c in string:
        optbytes[d] = ord(c)
        if d == len(string) + 1:
            if c == "/":
                optbytes[d] = 0
        d += 1

    return optbytes


def buildendoption():
    optbytes = bytearray(1)
    optbytes[0] = 255

    return optbytes


def decode_dhcp_request(packet):
    if len(packet) < 236:
        raise ValueError("Packet too short to be a DHCP packet")

    if packet[236:240] != MAGIC_COOKIE:
        raise ValueError("Packet does not contain magic cookie")

    # decode the DHCP packet and return a dictionary of the options
    # that were included in the packet
    # extract hops, transaction identfier, seconds and flags
    hops = packet[3]
    transactionid = packet[4:8]
    flags = packet[10:12]
    seconds = packet[8:10]

    # extract addresses
    ciaddr = packet[12:16]
    yiaddr = packet[16:20]
    siaddr = packet[20:24]
    giaddr = packet[24:28]

    # extract MAC address
    macaddr = packet[28:34]

    options = {}

    i = 240
    while i < len(packet):
        option = packet[i]
        if option == 0:
            i += 1
            continue

        if option == 255:
            break

        i += 1
        if i >= len(packet):
            logger.error(
                "Option %d - premature EOF when expecting length byte", option)
            break

        optlen = packet[i]
        i += 1

        if (i + optlen) >= len(packet):
            logger.error(
                "Option %d with length %d - premature EOF when "
                "expecting option data", option, optlen)
            break

        optdata = packet[i:i + optlen]
        options[option] = optdata

        i += optlen

    return transactionid, flags, macaddr, options


def dhcp_response(transactionid, flags, messagetype, baipaddr, baipbind,
                  thismacaddr, basubnet, bagateway, bootfilename):
    offer = bytearray(240)
    offer[0] = 2
    offer[1] = 1
    offer[2] = 6
    offer[3] = 0
    offer[4:8] = transactionid
    offer[10:12] = flags

    # assigned IP
    offer[16:20] = baipaddr

    # next server IP
    offer[20:24] = baipbind

    # put the MAC address in
    offer[28:34] = thismacaddr

    # insert DHCP cookie
    offer[236:240] = MAGIC_COOKIE

    # add options
    offer += build1byteoption(53, messagetype)

    offer += buildbyteoption(1, basubnet)  # Subnet mask
    if gateway != "":  # Gateway
        offer += buildbyteoption(3, bagateway)
    offer += build4byteoption(51, 0, 1, 81, 128)  # Lease time (24 hours)
    offer += buildbyteoption(54, baipbind)  # Server Identfier

    if bootfilename != "":  # Boot filename
        offer += buildstringoption(67, bootfilename)

    # terminate options
    offer += buildendoption()

    return offer


##############################################################################
def main(macaddr="",
         ipbind="",
         assigned_ip="",
         subnet="",
         gateway="",
         bootfilename=""):
    # check for good bind to IP addreress
    baipbind = socket.inet_aton(ipbind)
    basubnet = socket.inet_aton(subnet)
    bagateway = socket.inet_aton(gateway)

    # create a TCP/IP socket for receiving DHCP packets on port 67
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # bind the socket to the port
    server_address = (ipbind, 67)
    ### print("receive starting up on ", server_address)
    sock.bind(server_address)

    # main loop for DHCP server
    while True:
        # Wait for a connection
        packet, address = sock.recvfrom(32768)

        # decode the DHCP packet
        try:
            transactionid, flags, thismacaddr, opts = \
                decode_dhcp_request(packet)
        except ValueError as e:
            logger.error("Error decoding DHCP packet: %s", e)
            continue

        # see if there is a DHCP message type option
        if OPT_DHCP_MESSAGE_TYPE not in opts:
            continue

        messagetype = opts[OPT_DHCP_MESSAGE_TYPE]
        if (messagetype != b'\x01') and (messagetype != b'\x03'):
            print(
                "ignoring: DHCP message type not supported by this implementation",
                sep='')
            continue

        baipaddr = socket.inet_aton(assigned_ip)

        if messagetype == b'\01':
            # DHCPOFFER
            offer = dhcp_response(transactionid, flags, 2, baipaddr, baipbind,
                                  thismacaddr, basubnet, bagateway,
                                  bootfilename)
        elif messagetype == b'\03':
            # DHCPACK
            offer = dhcp_response(transactionid, flags, 5, baipaddr, baipbind,
                                  thismacaddr, basubnet, bagateway,
                                  bootfilename)
        else:
            continue

        # show the packet
        showpacket(offer)

        # send it
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sendsock:
            sendsock.bind((ipbind, 0))
            sendsock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            send_server_address = ('255.255.255.255', 68)
            sent = sendsock.sendto(offer, send_server_address)


if __name__ == '__main__':
    import sys

    numargs = len(sys.argv) - 1

    # if an odd number of arguments then something wrong
    if (numargs % 2) != 0:
        print("odd number of command line arguments", sep='')
        sys.exit()

    # set program defaults
    macaddr = ""
    ipbind = ""
    ipaddr = ""
    subnet = ""
    gateway = ""
    bootfilename = ""

    ### ipbind = "192.168.2.53"
    ### ipaddr = "192.168.2.100"
    ### subnet = "255.255.255.0"
    ### gateway = "192.168.2.254"

    # loop through command line args
    arg = 1
    while arg < numargs:
        if sys.argv[arg] == "-m":
            macaddr = (sys.argv[arg + 1]).upper()
        elif sys.argv[arg] == "-b":
            ipbind = sys.argv[arg + 1]
        elif sys.argv[arg] == "-i":
            ipaddr = sys.argv[arg + 1]
        elif sys.argv[arg] == "-s":
            subnet = sys.argv[arg + 1]
        elif sys.argv[arg] == "-g":
            gateway = sys.argv[arg + 1]
        elif sys.argv[arg] == "-f":
            bootfilename = sys.argv[arg + 1]
        else:
            print("unrecognised command line argument \"",
                  sys.argv[arg],
                  "\"",
                  sep='')
            exit()
        arg = arg + 2

    # ensure macaddr was set
    if macaddr == "":
        print("MAC address not specified with -m command line option", sep='')
        exit()

    # ensure IP bind address was set
    if ipbind == "":
        print(
            "interface IP address to bind to not specified with -b command line option",
            sep='')
        exit()

    main(macaddr, ipbind, ipaddr, subnet, gateway, bootfilename)
