# Auditoria técnica integral — Triton Kernel Lab

Data da auditoria: 21 de junho de 2026<br>
Hardware validado: NVIDIA GeForce RTX 4060 (SM89)<br>
Stack validada: Python 3.12.13, PyTorch 2.12.1+cu130, Triton 3.7.1

## Escopo e método

Foram revisados o código Python/Triton, fórmulas de forward e backward,
contratos de shape/dtype/stride, redução concorrente, benchmark harness,
persistência JSON, CI, Docker, documentação, testes e artefatos de resultados.

Arquivos locais de ferramentas pessoais sob `.claude/`, `.codex/` e `.agents/`
foram deliberadamente excluídos do produto.

# PARTE 1 — RELATÓRIO DE AUDITORIA

## Achados corrigidos

### 1. Oráculo numérico podia aprovar referência inválida

[Arquivo: `src/tklab/harness/tolerances.py`]<br>
[Linha(s): 68–84]<br>
[Gravidade] 🔴 Crítico<br>
[Problema] `assert_relative_frobenius` verificava apenas se a saída era finita.
Se a referência contivesse `NaN`, a norma relativa também virava `NaN`; a
comparação `NaN >= limite` é falsa e o teste terminava sem erro. Isso permitia
um falso positivo de corretude.<br>
[Solução Proposta] Rejeitar explicitamente referência não finita e erro relativo
não finito antes da comparação. Foram adicionados testes para `NaN`, `+inf` e
`-inf`.

### 2. Matmul aceitava stride incompatível com as premissas do kernel

[Arquivo: `src/tklab/kernels/matmul.py`]<br>
[Linha(s): 168–187]<br>
[Gravidade] 🔴 Crítico<br>
[Problema] O launcher aceitava tensores expandidos com stride zero, enquanto o
kernel declarava `tl.assume(stride > 0)`. Violar uma premissa de compilação pode
produzir acesso incorreto ou otimização inválida.<br>
[Solução Proposta] Rejeitar strides não estritamente positivos no contrato
público, manter o guard de endereço int32 e cobrir o caso com tensor `meta`.

### 3. Saída de vector-add podia ter aliasing e corridas de escrita

[Arquivo: `src/tklab/kernels/vector_add.py`]<br>
[Linha(s): 43–98]<br>
[Gravidade] 🔴 Crítico<br>
[Problema] A saída preservava o stride do primeiro operando. Se a entrada fosse
um `expand` com stride zero, todos os elementos de saída apontariam para o
mesmo endereço, causando escritas concorrentes e resultado semanticamente
incorreto.<br>
[Solução Proposta] Sempre alocar saída contígua, continuar aceitando leitura de
entradas com stride zero e validar a saída. Um teste real de GPU cobre 1.009
elementos com entrada expandida.

### 4. Evidência de sanitizer não cobria o backward novo da atenção

[Arquivo: `tests/test_sanitizer_workloads.py`]<br>
[Linha(s): 186–199]<br>
[Gravidade] 🟠 Alto<br>
[Problema] Os logs existentes exercitavam apenas o forward de Flash Attention.
Eles não provavam a indexação de `dQ`, `dK` e `dV` nas caudas de tile.<br>
[Solução Proposta] Adicionar workload causal e não causal em `N=129` e executar
`memcheck`, `initcheck`, `racecheck` e `synccheck`. Todos terminaram com zero
erros; `racecheck` registrou zero hazards e zero warnings. A evidência foi
separada em `compute_sanitizer_attention_backward_*.log`.

### 5. Ativações novas não tinham evidência de sanitizer

[Arquivo: `tests/test_sanitizer_workloads.py`]<br>
[Linha(s): 155–169]<br>
[Gravidade] 🟠 Alto<br>
[Problema] ReLU, GELU, SiLU e tanh tinham testes numéricos, mas não cobertura de
acesso, inicialização, sincronização e hazards.<br>
[Solução Proposta] Adicionar forward/backward com stride de linha independente
e cauda mascarada. As quatro ferramentas do Compute Sanitizer passaram sem
erros.

### 6. Cache de roofline podia ser reutilizado em ambiente incompatível

[Arquivo: `src/tklab/harness/roofline.py`]<br>
[Linha(s): 315–329, 565–649]<br>
[Gravidade] 🟠 Alto<br>
[Problema] Um `peaks.json` era aceito apenas pela versão de schema e presença de
alguns campos. Uma troca de GPU, compute capability, PyTorch ou Triton podia
reutilizar picos antigos e contaminar os percentuais publicados.<br>
[Solução Proposta] Comparar toda a proveniência ativa, exigir números positivos
e finitos e instruir recalibração com `--force-peaks` em qualquer divergência.

### 7. Schema de `peaks.json` não refletia a estrutura serializada

[Arquivo: `src/tklab/harness/roofline.py`]<br>
[Linha(s): 28, 565–632]<br>
[Gravidade] 🟠 Alto<br>
[Problema] A estrutura de `theoretical_provenance` havia mudado, mas o
discriminador permanecia na versão 3. O `TypedDict` e o JSON versionado
discordavam.<br>
[Solução Proposta] Elevar para schema 4, validar os campos aninhados e migrar
somente a proveniência derivada do `peaks.json`; nenhuma medição foi alterada.

### 8. LayerNorm perdeu o guard de endereço no gradiente upstream

[Arquivo: `src/tklab/kernels/layer_norm.py`]<br>
[Linha(s): 360–367]<br>
[Gravidade] 🟠 Alto<br>
[Problema] Após a centralização do guard int32, `dy` do LayerNorm não recebeu a
mesma verificação já presente no RMSNorm, SwiGLU e RoPE. Um layout extremo
poderia estourar a aritmética de offset.<br>
[Solução Proposta] Tornar `dy` contíguo quando necessário e aplicar
`assert_int32_addressable` antes do launch.

### 9. Forward público da atenção alocava estatística de backward em inferência

[Arquivo: `src/tklab/kernels/flash_attention.py`]<br>
[Linha(s): 630–663]<br>
[Gravidade] 🟠 Alto<br>
[Problema] Mesmo sem entradas com `requires_grad`, a API pública alocava e
preenchia o log-sum-exp `M`. Isso aumentava o footprint de inferência e
contaminaria o estudo de memória linear.<br>
[Solução Proposta] Despachar diretamente para `STORE_M=False` quando gradientes
estão desabilitados ou nenhuma entrada exige gradiente. O caminho de treino
continua salvando `M`.

### 10. Normalizações também guardavam estatísticas sem necessidade

[Arquivo: `src/tklab/kernels/layer_norm.py`]<br>
[Linha(s): 426–456]<br>
[Arquivo: `src/tklab/kernels/rms_norm.py`]<br>
[Linha(s): 424–443]<br>
[Arquivo: `src/tklab/kernels/residual_rms_norm.py`]<br>
[Linha(s): 230–267]<br>
[Gravidade] 🟡 Médio<br>
[Problema] Chamadas de inferência passavam pelo `autograd.Function` e alocavam
estatísticas exclusivas do backward.<br>
[Solução Proposta] Adicionar caminho sem estatísticas quando não há gradiente,
mantendo a mesma semântica e os launchers de benchmark.

### 11. Sigmoid estável estava duplicada e as ativações dependiam de contrato implícito

[Arquivo: `src/tklab/kernels/activations.py`]<br>
[Linha(s): 52–79]<br>
[Arquivo: `src/tklab/kernels/swiglu.py`]<br>
[Linha(s): 28–105]<br>
[Gravidade] 🟡 Médio<br>
[Problema] SwiGLU tinha uma implementação explícita resistente a overflow,
enquanto SiLU/tanh usavam outro caminho e duplicariam a política numérica.<br>
[Solução Proposta] Extrair `stable_sigmoid` para
`src/tklab/kernels/_elementwise_math.py` e reutilizar a fórmula por ramos
baseada em `exp(-abs(x))`.

### 12. Seeds das ativações não eram reproduzíveis entre processos

[Arquivo: `tests/test_activations.py`]<br>
[Linha(s): 27–34, 53, 67]<br>
[Gravidade] 🟡 Médio<br>
[Problema] `hash(name)` depende da randomização de hash do processo Python.
Falhas poderiam mudar de amostra a cada execução.<br>
[Solução Proposta] Substituir por seeds fixas e explícitas por ativação.

### 13. Epsilon inválido podia chegar aos kernels de normalização

[Arquivo: `src/tklab/kernels/_norm_common.py`]<br>
[Linha(s): 24–38]<br>
[Gravidade] 🟡 Médio<br>
[Problema] `eps=0`, negativo, infinito ou `NaN` podia gerar singularidade ou
estatísticas não finitas. A validação era repetida e não fazia parte do
contrato público.<br>
[Solução Proposta] Centralizar `validate_epsilon`, exigir valor finito e
estritamente positivo e testar LayerNorm, RMSNorm e residual RMSNorm.

### 14. Backwards customizados não declaravam o limite de primeira ordem

[Arquivo: `src/tklab/kernels/activations.py`]<br>
[Arquivo: `src/tklab/kernels/flash_attention.py`]<br>
[Arquivo: `src/tklab/kernels/layer_norm.py`]<br>
[Arquivo: `src/tklab/kernels/residual_rms_norm.py`]<br>
[Arquivo: `src/tklab/kernels/rms_norm.py`]<br>
[Arquivo: `src/tklab/kernels/rope.py`]<br>
[Arquivo: `src/tklab/kernels/swiglu.py`]<br>
[Linha(s): métodos `backward`]<br>
[Gravidade] 🟡 Médio<br>
[Problema] Os kernels retornam gradientes produzidos por Triton e não constroem
grafo para derivadas de ordem superior, mas esse limite não era explícito.<br>
[Solução Proposta] Aplicar `once_differentiable` e documentar suporte somente a
autograd de primeira ordem.

### 15. Contrato de `KernelSpec` dependia demais de type hints

[Arquivo: `src/tklab/registry.py`]<br>
[Linha(s): 103–151]<br>
[Gravidade] 🟡 Médio<br>
[Problema] Em runtime era possível fornecer `bound` ou `compute_mode`
desconhecido, dtypes duplicados, booleanos como tamanho e listas mutáveis dentro
de uma dataclass congelada.<br>
[Solução Proposta] Normalizar sequências para tuplas e validar tipos,
duplicatas, bounds, modos de computação e inteiros positivos.

### 16. Cadeia de build e CI continha referências mutáveis

[Arquivo: `Dockerfile`]<br>
[Linha(s): 20–22, 44–68]<br>
[Arquivo: `.github/workflows/ci.yml`]<br>
[Linha(s): 20–49]<br>
[Arquivo: `.pre-commit-config.yaml`]<br>
[Linha(s): 2–12]<br>
[Gravidade] 🟠 Alto<br>
[Problema] `uv:latest`, actions por tag major e comandos sem `--frozen`
permitiam mudanças externas sem alteração no repositório. O build context
também podia incluir configurações locais de agentes.<br>
[Solução Proposta] Fixar imagens por versão e digest, actions e hook por SHA,
Python 3.12.13, uv 0.11.23, usar `--frozen`, excluir configurações locais e
executar a imagem final como usuário sem privilégios.

### 17. Softmax e wrappers GPU tinham erros públicos instáveis

[Arquivo: `src/tklab/kernels/fused_softmax.py`]<br>
[Linha(s): 57–88, 101–113]<br>
[Arquivo: `src/tklab/kernels/vector_add.py`]<br>
[Linha(s): 56–74]<br>
[Arquivo: `src/tklab/kernels/matmul.py`]<br>
[Linha(s): 168–187]<br>
[Gravidade] 🟡 Médio<br>
[Problema] CPU comum podia chegar ao launcher Triton e falhar com erro interno;
o softmax também não validava o stride final da saída prealocada.<br>
[Solução Proposta] Rejeitar CPU com mensagem estável, preservando CPU somente
sob `TRITON_INTERPRET=1` para kernels compatíveis, e validar layout/endereço da
saída.

### 18. Backward da atenção dependia apenas da implementação SDPA como referência

[Arquivo: `tests/test_flash_attention_backward.py`]<br>
[Linha(s): 32–66, 98–111]<br>
[Gravidade] 🟡 Médio<br>
[Problema] Comparar somente com SDPA deixa uma possibilidade residual de erro
correlacionado de implementação ou backend.<br>
[Solução Proposta] Adicionar referência algébrica FP32 independente para
`dQ`, `dK` e `dV`. O erro relativo medido ficou aproximadamente entre
`2.85e-4` e `3.25e-4`, causal e não causal.

## Revisão matemática

- **LayerNorm:** `dx = rstd * (g - mean(g) - xhat * mean(g*xhat))`; `dweight`
  e `dbias` corretos. A redução por locks mantém count separado dos buffers.
- **RMSNorm:** `dx = rstd * (g - xhat * mean(g*xhat))`; ausência do termo
  `mean(g)` está correta porque não há centragem.
- **Residual RMSNorm:** a soma FP16 é materializada antes do upcast estatístico,
  preservando equivalência drop-in. `dx == dresidual` e o gradiente direto da
  saída residual é acumulado no backward.
- **SwiGLU:** `value * gate * sigmoid(gate)` e derivada
  `sigmoid(g) * (1 + g*(1-sigmoid(g)))`.
- **RoPE rotate-half:** forward e transposta/inversa no backward estão corretos;
  `dcos` e `dsin` seguem diretamente as duas componentes da rotação.
- **GELU exata:** `0.5*x*(1+erf(x/sqrt(2)))`; derivada
  `Phi(x) + x*phi(x)`.
- **Flash Attention:** online softmax, LSE base 2, `delta=sum(O*dO)`,
  `dS=P*(dP-delta)`, escala aplicada apenas em `dQ`/`dK`; máscaras causal e de
  cauda consistentes entre forward e backward.
- **Roofline RTX 4060:** perfil mantém 24 SMs, 4 Tensor Cores/SM e 128 FLOPs por
  clock por Tensor Core para FP16-input/FP32-accumulate, resultando em
  30,22848 TFLOP/s no boost nominal de 2.460 MHz.

## Arquivos revisados sem achados pendentes

Os arquivos abaixo foram revisados e não possuem achados pendentes no escopo
atual:

- `src/tklab/__init__.py`
- `src/tklab/cli.py`
- `src/tklab/harness/__init__.py`
- `src/tklab/harness/bench.py`
- `src/tklab/harness/jsonio.py`
- `src/tklab/harness/plots.py`
- `src/tklab/kernels/__init__.py`
- `benchmarks/__init__.py`
- `benchmarks/flash_attention_memory.py`
- `benchmarks/layer_norm_backward.py`
- `benchmarks/layer_norm_lock_stress.py`
- `benchmarks/profile_matmul.py`
- `benchmarks/rms_norm_lock_stress.py`
- `benchmarks/run.py`
- `tests/test_bench.py`
- `tests/test_cli.py`
- `tests/test_correctness.py`
- `tests/test_flash_attention.py`
- `tests/test_flash_attention_memory.py`
- `tests/test_jsonio.py`
- `tests/test_layer_norm.py`
- `tests/test_plots.py`
- `tests/test_residual_rms_norm.py`
- `tests/test_rms_norm.py`
- `tests/test_rope.py`
- `tests/test_swiglu.py`
- `.github/CODEOWNERS`
- `LICENSE`
- `SECURITY.md`
- `pyproject.toml`
- `uv.lock`
- `docs/benchmarking.md`
- `docs/code_review.md`
- `docs/fused_softmax.md`
- `docs/matmul.md`
- `docs/rope.md`
- `docs/swiglu.md`

Os PNGs são artefatos binários gerados e foram verificados quanto a presença,
tamanho e correspondência nominal com seus JSONs. Os JSONs históricos foram
mantidos; apenas `peaks.json` recebeu migração de schema/proveniência, sem
alteração dos valores medidos. Todos os logs de sanitizer existentes terminam
com resumo de zero erros.

## Limitações honestas, não defeitos

- CI hospedado não possui GPU; testes CUDA e sanitizers são evidência local.
- A cobertura CPU é 39%; código de dispositivo Triton não é observável pelo
  tracer Python e não foi removido artificialmente do denominador.
- O Dockerfile passou `docker build --check`; o build completo da imagem não
  foi executado nesta auditoria.
- Os novos kernels de ativação e os forwards de RMSNorm/residual RMSNorm ainda
  aguardam uma única sessão limpa de benchmark para gerar números publicáveis.
  Nenhum número contaminado foi inventado ou incluído.
- O projeto implementa autograd de primeira ordem, não derivadas superiores.

## Evidência de validação

- Ruff lint: aprovado.
- Ruff format: aprovado.
- mypy strict: aprovado em 51 arquivos Python.
- pytest CPU: 66 aprovados.
- pytest GPU: 83 aprovados.
- Triton interpreter: 2 aprovados, 13 skips esperados.
- Pre-commit completo: aprovado.
- Compute Sanitizer, ativações: quatro tools, zero erros/hazards/warnings.
- Compute Sanitizer, Flash Attention backward: quatro tools, zero
  erros/hazards/warnings.
- Docker build check: aprovado sem warnings.
- `git diff --check`: aprovado.

# PARTE 2 — CÓDIGO CORRIGIDO

O código corrigido está materializado no worktree, não apenas descrito neste
relatório. Os principais arquivos novos ou reescritos são:

```text
.dockerignore
.python-version
Dockerfile
docs/activations.md
docs/docker.md
src/tklab/harness/addressing.py
src/tklab/kernels/_elementwise_math.py
src/tklab/kernels/_norm_common.py
src/tklab/kernels/activations.py
tests/test_activations.py
tests/test_addressing.py
tests/test_flash_attention_backward.py
tests/test_vector_add.py
results/nvidia_geforce_rtx_4060/compute_sanitizer_activations_*.log
results/nvidia_geforce_rtx_4060/compute_sanitizer_attention_backward_*.log
```

Também foram corrigidos, no lugar, CI, Makefile, README, registry, harness,
todos os kernels afetados, testes de validação e documentação correspondente.
O diff do Git é a fonte canônica do código completo e evita que uma cópia
textual neste relatório diverja dos arquivos efetivamente testados.
