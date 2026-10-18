#ifndef CLOVER_DELIVER_H
#define CLOVER_DELIVER_H
/* Layer 93 posts the finished answer straight to the address the caller supplied, so
   nothing in the middle has to hold a connection open or carry a reply back upstream.

   The including file must define _GNU_SOURCE before its first include.

   The address comes from the caller, so this is an outbound request to somewhere this
   process does not control. It speaks http:// only, never follows a redirect and never
   reads more than a status line, which keeps the blast radius to one connection. */

#include <netdb.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

enum {
    CLOVER_DELIVER_HOST    = 256,
    CLOVER_DELIVER_PORT    = 16,
    CLOVER_DELIVER_PATH    = 1024,
    CLOVER_DELIVER_SECONDS = 10
};

static int clover_deliver_split(const char *url, char *host, char *port, char *path)
{
    const char *rest, *slash, *colon;
    size_t length;

    if (!url || strncmp(url, "http://", 7)) return 0;
    rest = url + 7;
    slash = strchr(rest, '/');
    length = slash ? (size_t)(slash - rest) : strlen(rest);
    if (!length || length >= CLOVER_DELIVER_HOST) return 0;

    colon = memchr(rest, ':', length);
    if (colon) {
        size_t host_length = (size_t)(colon - rest);
        size_t port_length = length - host_length - 1;
        if (!host_length || !port_length || port_length >= CLOVER_DELIVER_PORT) return 0;
        memcpy(host, rest, host_length); host[host_length] = 0;
        memcpy(port, colon + 1, port_length); port[port_length] = 0;
        if (strspn(port, "0123456789") != port_length) return 0;
    } else {
        memcpy(host, rest, length); host[length] = 0;
        memcpy(port, "80", 3);
    }
    if (!slash) { memcpy(path, "/", 2); return 1; }
    length = strlen(slash);
    if (length >= CLOVER_DELIVER_PATH) return 0;
    memcpy(path, slash, length + 1);
    return 1;
}

static int clover_deliver_all(int handle, const char *data, size_t length)
{
    while (length) {
        ssize_t sent = send(handle, data, length, MSG_NOSIGNAL);
        if (sent <= 0) return 0;
        data += sent;
        length -= (size_t)sent;
    }
    return 1;
}

/* Generated text is caller-visible data, so it is escaped rather than pasted. */
static size_t clover_deliver_escape(const char *text, size_t length, char *out, size_t capacity)
{
    size_t used = 0;
    for (size_t index = 0; index < length; index++) {
        unsigned char byte = (unsigned char)text[index];
        char piece[8];
        size_t width;
        if (byte == '"' || byte == '\\') { piece[0] = '\\'; piece[1] = (char)byte; width = 2; }
        else if (byte == '\n') { memcpy(piece, "\\n", 2); width = 2; }
        else if (byte == '\r') { memcpy(piece, "\\r", 2); width = 2; }
        else if (byte == '\t') { memcpy(piece, "\\t", 2); width = 2; }
        else if (byte < 0x20) { width = (size_t)snprintf(piece, sizeof piece, "\\u%04x", byte); }
        else { piece[0] = (char)byte; width = 1; }
        if (used + width >= capacity) return used;
        memcpy(out + used, piece, width);
        used += width;
    }
    return used;
}

static int clover_deliver(const char *url, const char *body, size_t length)
{
    char host[CLOVER_DELIVER_HOST], port[CLOVER_DELIVER_PORT], path[CLOVER_DELIVER_PATH];
    char header[CLOVER_DELIVER_PATH + CLOVER_DELIVER_HOST + 256], reply[128];
    struct addrinfo hints, *found = NULL, *step;
    struct timeval timeout;
    int handle = -1, header_length, delivered = 0;

    if (!body || !clover_deliver_split(url, host, port, path)) return 0;

    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, port, &hints, &found)) return 0;

    timeout.tv_sec = CLOVER_DELIVER_SECONDS;
    timeout.tv_usec = 0;
    for (step = found; step && handle < 0; step = step->ai_next) {
        handle = socket(step->ai_family, step->ai_socktype, step->ai_protocol);
        if (handle < 0) continue;
        (void)setsockopt(handle, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof timeout);
        (void)setsockopt(handle, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof timeout);
        if (!connect(handle, step->ai_addr, step->ai_addrlen)) break;
        close(handle);
        handle = -1;
    }
    freeaddrinfo(found);
    if (handle < 0) return 0;

    header_length = snprintf(header, sizeof header,
        "POST %s HTTP/1.1\r\nHost: %s\r\nContent-Type: application/json\r\n"
        "Content-Length: %zu\r\nConnection: close\r\n\r\n", path, host, length);
    if (header_length > 0 && (size_t)header_length < sizeof header &&
        clover_deliver_all(handle, header, (size_t)header_length) &&
        clover_deliver_all(handle, body, length)) {
        ssize_t got = recv(handle, reply, sizeof reply - 1, 0);
        if (got > 9) {
            reply[got] = 0;
            delivered = !strncmp(reply, "HTTP/1.", 7) && reply[9] == '2';
        }
    }
    close(handle);
    return delivered;
}

#endif
