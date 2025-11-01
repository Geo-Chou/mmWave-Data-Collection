import socket
import netifaces

class IPAddressFinder:
    @staticmethod
    def get_local_ip():
        """Returns the local IP address starting with 192.168, or None if not found."""
        for interface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(interface)
            if netifaces.AF_INET in addrs:
                for addr in addrs[netifaces.AF_INET]:
                    ip = addr['addr']
                    if ip.startswith("192.168"):
                        return ip
        return None  # Return None if no matching IP is found