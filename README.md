# 🧠 Bio Pixel IA

**Bio Pixel IA** é um projeto de inteligência artificial que estima a altura e o peso de um indivíduo com base em imagens frontais e laterais, utilizando proporções visuais do globo ocular em relação ao corpo. O projeto combina visão computacional, biometria e aprendizado de máquina para criar um sistema preciso e acessível.

---

## 📸 Objetivo

Desenvolver um pipeline que:
- Detecta olhos e landmarks faciais com precisão.
- Mede o globo ocular em pixels e estima sua escala real.
- Calcula a altura corporal com base na proporção ocular-corporal.
- Estima o peso corporal com base em medidas visuais da imagem lateral.

---

## 🧰 Tecnologias e Linguagens

### Linguagens principais
- **Python**: núcleo do projeto, usado para IA, visão computacional e análise de dados.
- **Markdown & YAML**: documentação e configuração de workflows.
- **JavaScript (opcional)**: para interface web interativa.
- **SQL (opcional)**: para armazenar dados e resultados.

### Bibliotecas e ferramentas
- `OpenCV` – processamento de imagem
- `MediaPipe` – detecção de landmarks faciais
- `Dlib` – detecção facial alternativa
- `Scikit-learn`, `XGBoost` – modelos de regressão
- `NumPy`, `Pandas` – manipulação de dados
- `Matplotlib`, `Seaborn` – visualização
- `TensorFlow`, `PyTorch` – redes neurais (opcional)
- `Jupyter Notebook` – prototipagem
- `Docker` – empacotamento do ambiente
- `GitHub Actions` – automação de testes e deploy

---

## 🧠 Áreas de Inteligência Artificial envolvidas

### 1. Visão Computacional
- Detecção de olhos e rosto
- Landmark facial
- Estimativa de pose
- Reconstrução 3D a partir de imagem 2D

### 2. Aprendizado de Máquina
- Regressão para estimar altura e peso
- Engenharia de atributos visuais
- Modelos supervisionados com dados antropométricos

### 3. Biometria e Antropometria Computacional
- Estudo de proporções corporais humanas
- Reconhecimento facial biométrico

### 4. Redes Neurais Convolucionais (CNNs)
- Detecção avançada de padrões visuais
- Estimativa de idade, sexo ou volume corporal

### 5. IA Multimodal (opcional)
- Combinação de imagem + texto para enriquecer estimativas

---

## 📋 Kanban do Projeto

### 🔮 Backlog
- Pesquisar datasets com anotações de olhos e altura/peso reais
- Estudar modelos de regressão para estimar peso com imagem lateral
- Testar precisão de MediaPipe vs Dlib
- Definir critérios mínimos de qualidade para imagens

### ✅ To Do
- Criar script para detectar olhos com MediaPipe
- Medir diâmetro da íris/globo ocular em pixels
- Medir altura corporal em pixels
- Converter proporção ocular-corporal em altura real
- Criar função para estimar peso com imagem lateral
- Montar pipeline de entrada para imagem frontal e lateral

### 🔧 In Progress
- Testes com imagens reais para validação da escala
- Ajuste fino da estimativa de peso

### ✅ Done
- Definição do objetivo do projeto
- Escolha da abordagem baseada em proporção ocular-corporal
- Identificação das ferramentas principais

---

## 📚 Fontes e Inspiração

- Estudos antropométricos (CAESAR, ANSUR II)
- Artigos sobre biometria ocular
- Projetos de IA multimodal e visão computacional

---

## 🚀 Contribuição

Sinta-se à vontade para abrir issues, enviar pull requests ou sugerir melhorias. Este projeto é uma exploração aberta da interseção entre IA, visão computacional e biometria.

---

## 📄 Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

---

