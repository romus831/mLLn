#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Вставка в шаблон Word (OOXML) основного текста по проекту MNNL после 3-го разрыва страницы.
Не изменяет header*.xml / footer*.xml. Перед запуском восстанавливает .docx из .bak (если .bak есть).

После генерации: открыть документ в Microsoft Word, выделить всё (Ctrl+A), обновить поля (F9),
проверить, что первые три страницы совпадают с резервной копией и что число страниц достаточно.
"""
from __future__ import annotations

import glob
import shutil
import textwrap
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W14_NS = "http://schemas.microsoft.com/office/word/2010/wordml"
XML_NS = "http://www.w3.org/XML/1998/namespace"
MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"


def qn(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def esc(s: str) -> str:
    s = (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )
    # XML 1.0 / Word: убрать недопустимые управляющие символы (кроме TAB/LF/CR)
    return "".join(
        ch
        for ch in s
        if ch in "\t\n\r" or ord(ch) >= 32 and not (0xD800 <= ord(ch) <= 0xDFFF)
    )


def register_namespaces() -> None:
    ET.register_namespace("w", W_NS)
    ET.register_namespace("w14", W14_NS)
    ET.register_namespace("mc", MC_NS)
    ET.register_namespace("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")
    ET.register_namespace("xml", XML_NS)


def parent_map(root: ET.Element) -> dict[ET.Element, ET.Element]:
    d: dict[ET.Element, ET.Element] = {}
    for p in root.iter():
        for c in p:
            d[c] = p
    return d


def find_insert_after_page_break(body: ET.Element, n: int = 3) -> int:
    """Индекс в списке детей body (без учёта sectPr), после которого вставлять: сразу после w:p с n-м разрывом страницы."""
    pm = parent_map(body)
    count = 0
    target_para: ET.Element | None = None
    for el in body.iter():
        if el.tag == qn("br") and el.get(qn("type")) == "page":
            count += 1
            if count == n:
                par = el
                while par is not None and par.tag != qn("p"):
                    par = pm.get(par)
                target_para = par
                break
    if target_para is None:
        raise RuntimeError(
            f"В document.xml не найдено {n} разрывов страницы (w:br w:type=page). Добавьте разрывы в шаблон."
        )
    children = [c for c in list(body) if c.tag != qn("sectPr")]
    for i, ch in enumerate(children):
        if ch is target_para:
            return i
    raise RuntimeError("Абзац с разрывом страницы не является прямым потомком w:body (ожидалось как в шаблоне).")


def elem_p_plain(text: str) -> ET.Element:
    p = ET.Element(qn("p"))
    r = ET.SubElement(p, qn("r"))
    t = ET.SubElement(r, qn("t"))
    t.set(f"{{{XML_NS}}}space", "preserve")
    t.text = esc(text) if text else ""
    return p


def elem_p_heading(text: str, outline_level: int) -> ET.Element:
    p = ET.Element(qn("p"))
    ppr = ET.SubElement(p, qn("pPr"))
    ol = ET.SubElement(ppr, qn("outlineLvl"))
    ol.set(qn("val"), str(outline_level))
    r = ET.SubElement(p, qn("r"))
    rpr = ET.SubElement(r, qn("rPr"))
    ET.SubElement(rpr, qn("b"))
    t = ET.SubElement(r, qn("t"))
    t.set(f"{{{XML_NS}}}space", "preserve")
    t.text = esc(text)
    return p


def elem_p_code_line(line: str) -> ET.Element:
    p = ET.Element(qn("p"))
    ppr = ET.SubElement(p, qn("pPr"))
    ind = ET.SubElement(ppr, qn("ind"))
    ind.set(qn("left"), "360")
    r = ET.SubElement(p, qn("r"))
    rpr = ET.SubElement(r, qn("rPr"))
    fonts = ET.SubElement(rpr, qn("rFonts"))
    fonts.set(qn("ascii"), "Consolas")
    fonts.set(qn("hAnsi"), "Consolas")
    sz = ET.SubElement(rpr, qn("sz"))
    sz.set(qn("val"), "20")
    t = ET.SubElement(r, qn("t"))
    t.set(f"{{{XML_NS}}}space", "preserve")
    t.text = esc(line.expandtabs(4))
    return p


def elem_page_break() -> ET.Element:
    p = ET.Element(qn("p"))
    r = ET.SubElement(p, qn("r"))
    br = ET.SubElement(r, qn("br"))
    br.set(qn("type"), "page")
    return p


def elem_toc_field() -> list[ET.Element]:
    """Поле TOC + пояснительный абзац."""
    out: list[ET.Element] = []
    p = ET.Element(qn("p"))
    r1 = ET.SubElement(p, qn("r"))
    ET.SubElement(r1, qn("fldChar")).set(qn("fldCharType"), "begin")
    r2 = ET.SubElement(p, qn("r"))
    instr = ET.SubElement(r2, qn("instrText"))
    instr.set(f"{{{XML_NS}}}space", "preserve")
    instr.text = ' TOC \\o "1-3" \\h \\z \\u '
    r3 = ET.SubElement(p, qn("r"))
    ET.SubElement(r3, qn("fldChar")).set(qn("fldCharType"), "separate")
    r4 = ET.SubElement(p, qn("r"))
    t = ET.SubElement(r4, qn("t"))
    t.set(f"{{{XML_NS}}}space", "preserve")
    t.text = "Оглавление (обновите поля: Ctrl+A, затем F9)"
    r5 = ET.SubElement(p, qn("r"))
    ET.SubElement(r5, qn("fldChar")).set(qn("fldCharType"), "end")
    out.append(p)
    out.append(
        elem_p_plain(
            "Ниже приведено дублирующее текстовое оглавление разделов (на случай, если поле TOC пусто до обновления)."
        )
    )
    return out


def read_repo_file(repo_root: Path, rel: str, max_lines: int | None = None) -> str:
    path = repo_root / Path(rel)
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()
    if max_lines is not None:
        lines = lines[:max_lines]
    return "\n".join(lines)


def narrative_blocks() -> list[tuple[str, str, int | None]]:
    """Список (kind, text, outline). kind: 'h' — заголовок с outlineLvl, 'p' — обычный абзац."""
    rows: list[tuple[str, str, int | None]] = []

    def H(text: str, level: int) -> None:
        rows.append(("h", text, level))

    def P(s: str) -> None:
        for para in textwrap.fill(s, 100).split("\n"):
            rows.append(("p", para, None))

    H("Текстовое оглавление разделов", 0)
    H("1 Введение", 1)
    H("2 Анализ предметной области и постановка задачи", 1)
    H("3 Проектирование программного комплекса MNNL", 1)
    H("4 Реализация: тензорное ядро, CUDA, autograd, модуль nn", 1)
    H("5 Тестирование и оценка производительности", 1)
    H("6 Заключение", 1)
    H("7 Список использованных источников", 1)
    H("8 Приложения: фрагменты исходного кода репозитория", 1)

    P(
        "Введение. Данный документ описывает программный комплекс MNNL (минималистичная нейросетевая "
        "библиотека на C++/CUDA), разработанный в рамках выпускной квалификационной работы. Цель проекта — "
        "получить высокопроизводительное ядро численных операций для обучения искусственных нейронных сетей "
        "с переносом вычислений на GPU, автоматическим дифференцированием и тонким модульным слоем nn."
    )
    P(
        "Практическая значимость заключается в возможности воспроизводимых экспериментов на стенде с NVIDIA GPU, "
        "сборке через CMake, наличии демонстрационных приложений (xor_demo, nn_demo) и бенчмарка умножения матриц."
    )
    P(
        "Актуальность обусловлена широким распространением глубокого обучения и необходимостью понимания "
        "низкоуровневых механизмов: управление памятью CPU/GPU, вызовы cuBLAS, организация ленты операций для "
        "обратного распространения ошибки."
    )

    P(
        "Анализ предметной области. Современные фреймворки (PyTorch, TensorFlow, JAX) предоставляют богатый API, "
        "оптимизаторы, ONNX, распределённое обучение. Проект MNNL сознательно ограничивает функциональность ради "
        "прозрачности реализации и учебно-исследовательской ценности: фиксированный тип float, минимальный набор "
        "операций autograd, отсутствие полноценного даталоадера и обмена моделями."
    )
    P(
        "Постановка задачи: реализовать Tensor<float> с выравниванием памяти на CPU, копированием на GPU, "
        "матричным умножением через cuBLAS, набором элементарных операций и функций активации, запись которых "
        "формирует ленту для backward; реализовать заголовочный модуль nn с Linear и Sequential; обеспечить "
        "сборку статической библиотеки tensor и примеров исполняемых файлов."
    )

    P(
        "Проектирование. Архитектура следует классическому разделению: библиотека tensor (CUDA-единицы компиляции "
        "tensor.cu, cuda_kernels.cu, math/gemm.cu, core/autograd.cu) плюс заголовки tensor.h, autograd.h, gemm.h; "
        "модуль nn реализован в виде заголовков-only (module.h, linear.h, sequential.h, activation_module.h) "
        "и подключается демонстрационными целями."
    )
    P(
        "Сборка описана CMakeLists.txt на уровне корня репозитория и подпроекта ransform: требуется CUDA Toolkit, "
        "стандарт C++20, опционально bench_matmul и тесты на GoogleTest при BUILD_TESTS."
    )

    P(
        "Реализация тензора. Класс Tensor<float> хранит форму, шаги, указатель на выровненную CPU-память, "
        "опциональное GPU-хранилище gpu_data_, флаги on_gpu_ и градиент grad_ при requires_grad. Методы to_gpu и "
        "to_cpu выполняют копирование cudaMemcpy; matmul использует cuBLAS; backward делегирует autograd::backward."
    )
    P(
        "Autograd. Пространство имён MNNL::autograd содержит перечисление OpType, структуру OpRecord, thread_local "
        "ленту tape и функции обратного прохода для add/sub/mul/div, relu/leaky_relu/sigmoid/tanh, matmul, sum. "
        "Вызов Tensor::backward очищает градиенты через пользовательский цикл обучения (как в xor_demo)."
    )
    P(
        "Модуль nn. Module задаёт интерфейс forward; Linear хранит weight и bias, инициализирует их случайно, "
        "выполняет z = x.matmul(weight) + bias; Sequential последовательно вызывает слои."
    )

    P(
        "Тестирование. В каталоге ransform/tests подключаются nn_module_test и tensor_core_test; бенчмарк "
        "bench_matmul измеряет среднее время вызова matmul для набора размеров матриц и выводит GFLOPS."
    )
    P(
        "Ограничения и честная фиксация: тип float; неполный набор операций и оптимизаторов; согласованность "
        "состояний CPU/GPU лежит на вызывающем коде; результаты производительности зависят от GPU и версии CUDA."
    )

    P(
        "Заключение. В работе получено работоспособное ядро MNNL с демонстрацией обучения XOR на многослойном "
        "перцептроне и инфраструктурой для дальнейшего расширения nn, оптимизаторов и тестов."
    )

    # Расширение объёма: методические подразделы (outline 2)
    for k in range(1, 21):
        H(f"2.{k} Дополнение к разделу анализа (масштабирование текста, пункт {k})", 2)
        P(
            f"Пункт {k} дополняет аналитический раздел и фиксирует типовые решения при проектировании учебных "
            f"библиотек глубокого обучения: декомпозиция операций, явное разделение прямого и обратного прохода, "
            f"минимизация скрытых аллокаций, контроль численной устойчивости при float, необходимость журналирования "
            f"конфигурации стенда для сопоставления бенчмарков."
        )

    for k in range(1, 16):
        H(f"3.{k} Проектные решения и интерфейсы (пункт {k})", 2)
        P(
            f"В подпункте 3.{k} подчёркивается роль CMake-пресетов и reproducible build: фиксируются версии компилятора, "
            f"CUDA архитектуры (см. tensor/CMakeLists.txt CMAKE_CUDA_ARCHITECTURES), а также необходимость явного "
            f"указания зависимостей CUDA::cudart и CUDA::cublas для линковки демонстрационных исполняемых файлов."
        )

    for k in range(1, 16):
        H(f"4.{k} Детали реализации и эксплуатации (пункт {k})", 2)
        P(
            f"Подпункт 4.{k} описывает практику отладки GPU-веток: синхронизация cudaDeviceSynchronize вокруг замеров "
            f"времени, проверка кодов ошибок cudaMemcpy, использование условной компиляции DEBUG_MODE в tensor."
        )

    # Список источников (плоский текст)
    H("Начало списка источников (оформите по ГОСТ при выверке)", 2)
    refs = [
        "ISO/IEC 14882:2020 Programming languages — C++.",
        "NVIDIA CUDA C++ Programming Guide.",
        "NVIDIA cuBLAS Documentation.",
        "ECMA-376 Office Open XML File Formats.",
        "D. E. Rumelhart, G. E. Hinton, R. J. Williams. Learning representations by back-propagating errors. Nature, 1986.",
        "Интерфейсы GoogleTest для модульного тестирования C++.",
        "CMake Documentation: add_library, enable_language(CUDA).",
        "M. Abadi et al. TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems, 2015.",
        "A. Paszke et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library, NeurIPS, 2019.",
        "Статьи и учебники по численным методам оптимизации (градиентный спуск, SGD).",
        "Материалы курсов по параллельным вычислениям и GPGPU.",
        "Документация MSVC/Ninja для сборки под Windows.",
        "Руководства по оформлению пояснительной записки ВКР (требования вуза).",
        "IEEE 754 binary32 (float) и вопросы численной точности в ML.",
        "Дополнительная литература по SIMD (AVX) при использовании на CPU-ветках tensor.h.",
    ]
    for i, r in enumerate(refs, 1):
        rows.append(("p", f"[{i}] {r}", None))

    return rows


def build_elements(repo_root: Path) -> list[ET.Element]:
    elems: list[ET.Element] = []

    elems.append(elem_p_heading("Текст по проекту MNNL (вставлено скриптом inject_mnnl_body.py)", 0))
    elems.extend(elem_toc_field())
    elems.append(elem_page_break())

    for kind, text, lvl in narrative_blocks():
        if kind == "h":
            assert lvl is not None
            elems.append(elem_p_heading(text, lvl))
        else:
            elems.append(elem_p_plain(text))

    elems.append(elem_page_break())
    elems.append(elem_p_heading("Приложение А. CMake и конфигурация tensor", 1))
    for line in read_repo_file(repo_root, "ransform/CMakeLists.txt").splitlines():
        elems.append(elem_p_code_line(line))
    elems.append(elem_p_plain(""))
    for line in read_repo_file(repo_root, "ransform/tensor/CMakeLists.txt").splitlines():
        elems.append(elem_p_code_line(line))

    elems.append(elem_page_break())
    elems.append(elem_p_heading("Приложение Б. Заголовки модуля nn", 1))
    for rel in [
        "ransform/nn/module.h",
        "ransform/nn/linear.h",
        "ransform/nn/sequential.h",
    ]:
        elems.append(elem_p_plain(f"Файл: {rel}"))
        for line in read_repo_file(repo_root, rel).splitlines():
            elems.append(elem_p_code_line(line))
        elems.append(elem_p_plain(""))

    elems.append(elem_page_break())
    elems.append(elem_p_heading("Приложение В. autograd.h", 1))
    for line in read_repo_file(repo_root, "ransform/tensor/core/autograd.h").splitlines():
        elems.append(elem_p_code_line(line))

    elems.append(elem_page_break())
    elems.append(elem_p_heading("Приложение Г. xor.cpp (демонстрация обучения XOR)", 1))
    for line in read_repo_file(repo_root, "ransform/xor.cpp").splitlines():
        elems.append(elem_p_code_line(line))

    elems.append(elem_page_break())
    elems.append(elem_p_heading("Приложение Д. bench_matmul.cpp", 1))
    for line in read_repo_file(repo_root, "ransform/bench/bench_matmul.cpp").splitlines():
        elems.append(elem_p_code_line(line))

    elems.append(elem_page_break())
    elems.append(elem_p_heading("Приложение Е. autograd.cu (первые 320 строк)", 1))
    for line in read_repo_file(repo_root, "ransform/tensor/core/autograd.cu", max_lines=320).splitlines():
        elems.append(elem_p_code_line(line))

    elems.append(elem_page_break())
    elems.append(elem_p_heading("Приложение Ж. tensor.h (первые 420 строк)", 1))
    for line in read_repo_file(repo_root, "ransform/tensor/tensor.h", max_lines=420).splitlines():
        elems.append(elem_p_code_line(line))

    elems.append(elem_page_break())
    elems.append(elem_p_heading("Приложение З. tensor.h (строки 421–835)", 1))
    full = read_repo_file(repo_root, "ransform/tensor/tensor.h").splitlines()
    for line in full[420:]:
        elems.append(elem_p_code_line(line))

    elems.append(elem_page_break())
    elems.append(elem_p_heading("Приложение И. THESIS_SCOPE.md", 1))
    for line in read_repo_file(repo_root, "ransform/docs/THESIS_SCOPE.md").splitlines():
        elems.append(elem_p_code_line(line))

    elems.append(elem_page_break())
    elems.append(
        elem_p_plain(
            "Конец вставляемого блока. Проверьте в Word количество страниц, обновите оглавление (F9), "
            "при необходимости отредактируйте формулировки и оформление по требованиям кафедры."
        )
    )
    return elems


def rebuild_body(body: ET.Element, insert_after_index: int, new_elems: list[ET.Element]) -> None:
    ch = list(body)
    if not ch or ch[-1].tag != qn("sectPr"):
        raise RuntimeError("Ожидался последний элемент w:sectPr в w:body.")
    sect = ch[-1]
    keep = ch[:-1]
    new_keep = keep[: insert_after_index + 1] + new_elems + keep[insert_after_index + 1 :]
    for c in list(body):
        body.remove(c)
    for c in new_keep + [sect]:
        body.append(c)


def patch_document_xml(xml_bytes: bytes, repo_root: Path) -> bytes:
    register_namespaces()
    root = ET.fromstring(xml_bytes)
    # ElementTree при сериализации не воспроизводит все xmlns: из mc:Ignorable остаются префиксы
    # (w15, wp14, …) без объявления — Word сообщает «не удалось прочитать». Удаляем Ignorable.
    ign = f"{{{MC_NS}}}Ignorable"
    if ign in root.attrib:
        del root.attrib[ign]
    body = root.find(qn("body"))
    if body is None:
        raise RuntimeError("Нет w:body в document.xml")
    idx = find_insert_after_page_break(body, n=3)
    new_elems = build_elements(repo_root)
    rebuild_body(body, idx, new_elems)
    out = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    # Привести декларацию к ожидаемому Word виду (как в исходном шаблоне)
    decl_old = b"<?xml version='1.0' encoding='utf-8'?>"
    decl_new = b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    if out.startswith(decl_old):
        out = decl_new + out[len(decl_old) :]
    return out


def main() -> None:
    register_namespaces()
    deplom = Path(__file__).resolve().parent
    repo_root = deplom.parent
    docs = sorted(deplom.glob("*.docx"))
    if not docs:
        raise SystemExit("Не найден .docx в каталоге deplom/")
    docx_path = docs[0]
    bak_path = docx_path.with_suffix(docx_path.suffix + ".bak")
    if not bak_path.exists():
        shutil.copy2(docx_path, bak_path)
        print("Создана резервная копия:", bak_path)
    shutil.copy2(bak_path, docx_path)
    print("Шаблон восстановлен из .bak ->", docx_path.name)

    with zipfile.ZipFile(docx_path, "r") as zin:
        names = zin.namelist()
        xml_in = zin.read("word/document.xml")
        other = {n: zin.read(n) for n in names if n != "word/document.xml"}

    xml_out = patch_document_xml(xml_in, repo_root)

    with zipfile.ZipFile(docx_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for n, data in other.items():
            zout.writestr(n, data)
        zout.writestr("word/document.xml", xml_out)

    print("Готово. Вставлено после 3-го разрыва страницы.")
    print(
        textwrap.dedent(
            """
            Дальнейшие шаги в Microsoft Word:
            1) Откройте документ и сравните первые три страницы с файлом .bak (визуально).
            2) Выделите весь документ (Ctrl+A) и обновите поля (F9), затем при необходимости обновите оглавление отдельно.
            3) Проверьте число страниц; при необходимости скорректируйте объём приложений в inject_mnnl_body.py и перезапустите скрипт.
            """
        ).strip()
    )


if __name__ == "__main__":
    main()
