; Sieve of Eratosthenes over cells [base, base+limit).
; After it runs, mem[base + k] == 1 exactly when k is composite (k >= 2).
let limit = 1
let base  = 2
let i     = 3
let j     = 4
let p     = 5
let flag  = 6
let end   = 7
  add end, base, limit
  mov i, #2
outer:
  add p, base, i          ; p = &mem[base + i]
  lt  flag, p, end        ; still inside the table?
  jz  flag, done
  load flag, p            ; already marked composite?
  jnz flag, next
  add j, i, i             ; j = 2i
inner:
  add p, base, j
  lt  flag, p, end
  jz  flag, next
  store p, #1
  add j, j, i
  jmp inner
next:
  add i, i, #1
  jmp outer
done:
  halt
