# 🧠 Bio Pixel IA

**Bio Pixel IA** é um projeto de IA e Visão Computacional que estima a **altura** de **múltiplos indivíduos** em uma imagem frontal. O sistema evoluiu de um simples script para uma **aplicação web Django** que calcula a altura usando um sistema hierárquico de biometria facial e gera um **Score de Confiança** para cada medição.

O núcleo do projeto é um pipeline de "dupla-verificação" que usa:
1.  **Plano A (Padrão Ouro):** O diâmetro da Íris (`~12mm`), uma constante biológica.
2.  **Plano B (Padrão Prata):** A Distância Interpupilar (IPD) (`~63mm`), uma média estatística usada como *fallback*.

O sistema combina essas medidas com validações de pose, profundidade de lente (Z-score) e qualidade da imagem para estimar a altura com a maior precisão possível.

---

## 📸 Objetivos Atuais do Pipeline

O pipeline processa um upload de imagem e executa as seguintes etapas:
- **Detecção de Corpos:** Utiliza **YOLOv8-Pose** para detectar todas as pessoas na imagem e seus principais landmarks corporais (cabeça, calcanhares).
- **Medição de Altura (Pixels):** Calcula a altura de cada pessoa em pixels.
- **Detecção de Rosto:** Isola o rosto de cada pessoa detectada.
- **Análise Facial 3D:** Executa o **MediaPipe FaceMesh** no recorte do rosto para obter landmarks 3D (X, Y, Z).
- **Validação de Qualidade:**
    - **Pose Frontal:** Verifica se o rosto está virado (comparando distâncias X/Y).
    - **Profundidade (Lente):** Verifica se a perspectiva da lente está distorcida (analisando o Z-score da íris).
- **Cálculo de Escala Dupla (A Lógica Central):**
    1.  Tenta o **Plano A (Íris)**, validando a qualidade da medição (se é um círculo, se há óculos, etc.).
    2.  Tenta o **Plano B (IPD)** como uma medida robusta.
- **Cálculo de Confiança:**
    - Gera um **Score de Confiança (0-100%)** que é penalizado por má pose, profundidade ruim ou inconsistências.
    - O score recebe um **bônus** se as escalas da Íris e do IPD forem muito próximas.
- **Estimativa Final:** Calcula a altura em `cm` usando a escala de maior confiança.
- **Recursos Adicionais:** Estima idade (usando um modelo `ONNX`), cor dos olhos e cor do cabelo.
- **Interface:** Exibe todos os resultados por pessoa em uma interface web **Django**.

---

## 🧰 Tecnologias e Frameworks

### Linguagens e Frameworks
- **Python**: Núcleo de todo o processamento.
- **Django**: Framework web para a interface do usuário, uploads e processamento.
- **Ultralytics (YOLO)**: Framework de detecção de objetos/pose.
- **SQL (via SQLite)**: Banco de dados padrão do Django.
- **HTML/CSS**: Para o template `upload.html`.

### Bibliotecas de IA e Visão Computacional
- `OpenCV` – Processamento, leitura e escrita de imagens.
- `MediaPipe` – Detecção de landmarks faciais 3D (FaceMesh).
- `ONNX Runtime` – Execução do modelo de estimativa de idade.
- `NumPy` – Todos os cálculos numéricos e de vetores.

### Ferramentas de Desenvolvimento
- `Git & GitHub` – Controle de versão.
- `venv` (Python 3.10) – Gerenciamento de ambiente.

---

## 📋 Kanban do Projeto

### 🔮 Backlog / Próximos Passos

- **O GRANDE SALTO: Loop de Feedback (Active Learning)**
    - *Ideia:* Permitir que o usuário insira a **altura real** após a estimativa.
    - *Ação:*
        1.  Criar um `models.py` no Django para salvar *todas* as métricas (altura_pixels, escala_iris, escala_ipd, pose_diff, z_depth, altura_estimada, altura_real).
        2.  Criar uma nova view e URL (`/salvar_feedback/<id>`) para salvar a altura real enviada pelo usuário.
        3.  Criar um script `train_model.py` que use Scikit-learn/XGBoost para treinar um modelo de regressão (`X` = todas as métricas, `y` = altura_real).
        4.  O sistema de "auto-ajuste" aprenderá com os dados e, eventualmente, o `views.py` usará `model.predict()` em vez da nossa heurística atual.

- **Implementar Estimativa de Peso (Objetivo Original)**
    - Requer um novo formulário de upload para uma **imagem lateral**.
    - Pesquisar modelos ou heurísticas que correlacionem área de superfície/volume visível com o peso.

- **Ajuste Fino do `CALIBRACAO_ESCALA`**
    - Com base nos dados do "Loop de Feedback", podemos encontrar um fator de calibração mais preciso.

### 🔧 In Progress
- Testes de robustez com diferentes tipos de óculos (reflexos, armações grossas).
- Ajuste fino dos pesos do "Score de Confiança" (ex: qual a penalidade ideal para uma pose ruim?).

### ✅ Done (Conquistas Recentes)
- **Migração para Django:** O projeto agora é uma aplicação web completa.
- **Detecção Multi-Pessoa:** Substituído o `MediaPipe Pose` (single-person) pelo **YOLOv8-Pose** (multi-person).
- **Pipeline de Dupla Escala:** Criado o pipeline de **Plano A (Íris)** e **Plano B (IPD)** para cálculo de escala.
- **Validação Avançada:** Implementados 3 níveis de validação: **Pose Frontal** (X/Y), **Profundidade da Lente** (Z-score) e **Qualidade da Íris** (desvio padrão).
- **Score de Confiança:** Implementada sua ideia de gerar um score de confiança baseado na consistência das medidas.
- **Integração de Modelo:** Modelo `ONNX` de estimativa de idade integrado com sucesso (`age_utils`).
- **Limpeza de Repositório:** O repositório Git foi totalmente limpo (remoção de `venv`, `db.sqlite3`, `media/`) e o histórico foi reescrito com `git rebase` para remover commits "sujos".
- **Ambiente Estável:** O ambiente `venv310` (Python 3.10) está estável e o `requirements.txt` foi corrigido (resolvendo o erro `ResolutionImpossible`).

---

## 🚀 Contribuição

Sinta-se à vontade para abrir issues, enviar pull requests ou sugerir melhorias. Este projeto é uma exploração aberta da interseção entre IA, visão computacional e biometria.

---

## 📄 Licença

Este projeto está licenciado sob a [MIT License](LICENSE).