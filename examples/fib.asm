; fib: leaves fib(n) (mod 1024) in `b`
let n = 1
let a = 2
let b = 3
let t = 4
  mov a, #1
  mov b, #0
loop:
  jz  n, done
  add t, a, b
  mov a, b
  mov b, t
  sub n, n, #1
  jmp loop
done:
  halt
