---
name: flash-device
description: Use when flashing a kws-de build onto the CoreS3, talking to its serial console, or about to hand-type esptool flags -- the device is plugged into a remote host over ssh, not this machine.
---

# Flashing and talking to the CoreS3

The CoreS3 is USB-attached to a remote host (`$KWSREC_HOST`), never to the
machine Claude Code runs on. Every flash and every serial read happens over
`ssh $KWSREC_HOST`.

## Facts (don't rediscover)

- **Serial port** is the ESP32-S3's own USB-Serial-JTAG (`/dev/cu.usbmodem*`
  on the host); the CoreS3 has no UART bridge, so nothing reads UART0/stdin.
- **Opening the port resets the chip** (DTR/RTS toggle on open). A plain
  `cat`/`stty`/default-pyserial open can leave it silent for good --
  `console.sh` (below) opens with both lines held low and waits out the
  boot log first.
- **USB-drive mode**: the JTAG port disappears; a **CDC-ACM port with a
  different device node** carries the console instead. `ls
  /dev/cu.usbmodem*` again to find it, then send `mode menu` over it to
  leave USB mode -- that CDC port cannot be used to flash.
- **One command per tool invocation.** Never chain two `console.sh`/
  `flash.sh` calls in one Bash call (`&&`, `;`): Claude Code's tool runner
  dispatches chained commands only ~30 ms apart, well before the previous
  port open's reset and boot log settle. Issue one, read its output, then
  issue the next.
- Console protocol: one line per command --
  `mode menu|record|recordwake|recognise|wake|usb`, `status` -- every reply
  ends with `ok`/`err ...`.

## Flashing

```bash
KWSREC_HOST=<host> scripts/flash.sh [kws-de-fw-<sha>.bin] [port]
```

Copies the image to the host and runs `esptool.py` there. `KWSREC_PORT`
overrides port autodetection; without it, `flash.sh` lists
`/dev/cu.usbmodem*` on the host itself. Reflashing only ever writes the app
partition -- recordings on the device's `storage` FAT partition survive.
`flash.sh` does not itself capture the boot log; read it afterwards with
`console.sh` (below, own invocation) rather than guessing whether boot
completed.

Without `KWSREC_HOST` set, `flash.sh` flashes a locally-attached device
instead -- useful only when the CoreS3 is plugged into the machine running
Claude Code, which is not the usual case for this project.

## Talking to the console

```bash
.claude/skills/flash-device/console.sh -H <host> status
.claude/skills/flash-device/console.sh -H <host> -l 45 'mode wake'   # 45 s of the wake peak trace
.claude/skills/flash-device/console.sh 'mode menu'                  # $KWSREC_HOST already set
```

`-w SECONDS` (default 3) is how long it waits after opening the port for the
boot log to pass before sending the command; raise it if a reply comes back
empty. `-l SECONDS` (default 5) is how long it keeps reading after sending
the command -- raise it for a trace that takes longer to print (e.g. a wake
peak trace, or capturing a UI screenshot frame, which is ~4 s of base64 at
115200 baud). Exactly one command per invocation -- see the 30 ms rule above.

## Common mistakes

| Mistake | Fix |
|---|---|
| Two `console.sh`/`flash.sh` calls chained with `&&` | One per tool invocation; the chip has not finished resetting from the first |
| `cat`/`stty`/plain pyserial to read or send -> 0 bytes | Use `console.sh` (DTR/RTS low, waits out the boot log) |
| Port missing | Device may be in USB-drive mode (serial link gone) -- tap Back/Record on the device, or unplug/replug, then retry |
| Assuming `flash.sh` talks to a remote host by default | Set `KWSREC_HOST`; unset, it flashes locally |
| Capturing too few seconds of a screen-frame or wake trace | Raise `-l`; a screen frame alone is ~4 s of base64 |
