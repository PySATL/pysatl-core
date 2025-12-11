#!/usr/bin/env python3
"""
Продвинутый пример использования UNURAN через CFFI с callback функциями.

Этот пример показывает, как:
1. Создавать C callback функции из Python функций
2. Устанавливать PDF/CDF через callback
3. Генерировать реальные выборки
"""

import sys
from pathlib import Path
import math

# Добавляем путь к исходникам
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))


def example_with_pdf_callback():
    """
    Пример с callback функцией для PDF.
    
    Создаем простое распределение с известной PDF функцией.
    """
    print("=" * 70)
    print("ПРИМЕР: Генерация выборок с callback функцией PDF")
    print("=" * 70)
    
    try:
        from pysatl_core.stats._unuran.bindings import _unuran_cffi
        
        ffi = _unuran_cffi.ffi
        lib = _unuran_cffi.lib
        
        # ВАЖНО: храним ссылки на callback функции, чтобы предотвратить сборку мусора
        callbacks = []
        
        # Определяем PDF функцию (например, простая экспоненциальная)
        # PDF(x) = lambda * exp(-lambda * x) для x >= 0
        lambda_param = 1.0
        
        def pdf_func(x, distr_ptr):
            """PDF функция для экспоненциального распределения"""
            if x < 0:
                return 0.0
            return lambda_param * math.exp(-lambda_param * x)
        
        def dpdf_func(x, distr_ptr):
            """Производная PDF для экспоненциального распределения"""
            if x < 0:
                return 0.0
            # d/dx [lambda * exp(-lambda * x)] = -lambda^2 * exp(-lambda * x)
            return -lambda_param * lambda_param * math.exp(-lambda_param * x)
        
        # Создаем C callback функции
        print("\n1. Создание C callback функций для PDF и dPDF...")
        pdf_callback = ffi.callback(
            "double(double, const struct unur_distr*)",
            pdf_func
        )
        callbacks.append(pdf_callback)  # Сохраняем ссылку
        
        dpdf_callback = ffi.callback(
            "double(double, const struct unur_distr*)",
            dpdf_func
        )
        callbacks.append(dpdf_callback)  # Сохраняем ссылку
        
        print("   ✓ Callback функции созданы")
        
        # Создаем распределение
        print("\n2. Создание распределения...")
        distr = lib.unur_distr_cont_new()
        if distr == ffi.NULL:
            print("   ✗ Не удалось создать распределение")
            return False
        print("   ✓ Распределение создано")
        
        # Устанавливаем область определения [0, infinity)
        print("\n3. Настройка распределения...")
        lib.unur_distr_cont_set_domain(distr, 0.0, ffi.cast("double", float('inf')))
        print("   ✓ Область определения установлена [0, +inf)")
        
        # Устанавливаем PDF и производную PDF через callback
        print("\n4. Установка PDF и производной PDF через callback...")
        result = lib.unur_distr_cont_set_pdf(distr, pdf_callback)
        if result != 0:
            print(f"   ⚠ Код возврата при установке PDF: {result}")
        else:
            print("   ✓ PDF установлена")
        
        result = lib.unur_distr_cont_set_dpdf(distr, dpdf_callback)
        if result != 0:
            print(f"   ⚠ Код возврата при установке dPDF: {result}")
        else:
            print("   ✓ Производная PDF (dPDF) установлена")
        
        # Создаем параметры метода AROU (требует PDF и dPDF)
        print("\n5. Создание параметров метода AROU...")
        par = lib.unur_arou_new(distr)
        if par == ffi.NULL:
            errno = lib.unur_get_errno()
            print(f"   ✗ Не удалось создать параметры (errno={errno})")
            if errno != 0:
                error_str = lib.unur_get_strerror(errno)
                if error_str:
                    print(f"   Описание: {ffi.string(error_str).decode('utf-8')}")
            lib.unur_distr_free(distr)
            return False
        print("   ✓ Параметры созданы")
        
        # Инициализируем генератор
        print("\n6. Инициализация генератора...")
        gen = lib.unur_init(par)
        if gen == ffi.NULL:
            errno = lib.unur_get_errno()
            print(f"   ✗ Не удалось инициализировать генератор (errno={errno})")
            if errno != 0:
                error_str = lib.unur_get_strerror(errno)
                if error_str:
                    print(f"   Описание: {ffi.string(error_str).decode('utf-8')}")
            lib.unur_par_free(par)
            lib.unur_distr_free(distr)
            return False
        print("   ✓ Генератор инициализирован")
        
        # Генерируем выборки
        print("\n7. Генерация выборок...")
        n_samples = 10
        samples = []
        
        try:
            for i in range(n_samples):
                sample = lib.unur_sample_cont(gen)
                samples.append(sample)
                print(f"   Sample {i+1:2d}: {sample:10.6f}")
            
            print(f"\n   ✓ Успешно сгенерировано {len(samples)} выборок")
            
            # Статистика
            if samples:
                mean = sum(samples) / len(samples)
                print(f"\n   Статистика:")
                print(f"   Среднее: {mean:.6f} (ожидаемое: {1.0/lambda_param:.6f})")
                print(f"   Минимум: {min(samples):.6f}")
                print(f"   Максимум: {max(samples):.6f}")
        
        except Exception as e:
            print(f"   ✗ Ошибка при генерации: {e}")
            import traceback
            traceback.print_exc()
        
        # Освобождаем память
        print("\n8. Освобождение памяти...")
        lib.unur_free(gen)
        # ВАЖНО: после unur_init(par) параметры автоматически освобождаются
        # Не нужно вызывать unur_par_free(par) - это может привести к двойному освобождению
        lib.unur_distr_free(distr)
        print("   ✓ Память освобождена")
        
        # Важно: callback функции должны оставаться в памяти до освобождения всех объектов
        # Мы сохранили ссылки в списке callbacks, чтобы предотвратить сборку мусора
        # После освобождения distr можно безопасно удалить ссылки
        
        return len(samples) > 0
        
    except ImportError as e:
        print(f"\n✗ Ошибка импорта: {e}")
        print("\nУбедитесь, что модуль скомпилирован и доступен.")
        return False
    except Exception as e:
        print(f"\n✗ Ошибка: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def example_with_pdf_and_dpdf():
    """
    Пример с PDF и производной PDF (для более эффективных методов).
    """
    print("\n" + "=" * 70)
    print("ПРИМЕР: Генерация с PDF и производной PDF")
    print("=" * 70)
    
    try:
        from pysatl_core.stats._unuran.bindings import _unuran_cffi
        
        ffi = _unuran_cffi.ffi
        lib = _unuran_cffi.lib
        
        # ВАЖНО: храним ссылки на callback функции, чтобы предотвратить сборку мусора
        callbacks = []
        
        # Определяем PDF и производную PDF для нормального распределения
        # PDF(x) = (1/sqrt(2*pi)) * exp(-x^2/2)
        # dPDF/dx = -x * PDF(x)
        
        def pdf_func(x, distr_ptr):
            """PDF стандартного нормального распределения"""
            return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)
        
        def dpdf_func(x, distr_ptr):
            """Производная PDF стандартного нормального распределения"""
            pdf_val = pdf_func(x, distr_ptr)
            return -x * pdf_val
        
        # Создаем callback функции
        print("\n1. Создание callback функций...")
        pdf_callback = ffi.callback(
            "double(double, const struct unur_distr*)",
            pdf_func
        )
        callbacks.append(pdf_callback)  # Сохраняем ссылку
        
        dpdf_callback = ffi.callback(
            "double(double, const struct unur_distr*)",
            dpdf_func
        )
        callbacks.append(dpdf_callback)  # Сохраняем ссылку
        
        print("   ✓ Callback функции созданы")
        
        # Создаем распределение
        print("\n2. Создание и настройка распределения...")
        distr = lib.unur_distr_cont_new()
        if distr == ffi.NULL:
            print("   ✗ Не удалось создать распределение")
            return False
        
        lib.unur_distr_cont_set_domain(distr, -5.0, 5.0)
        lib.unur_distr_cont_set_pdf(distr, pdf_callback)
        lib.unur_distr_cont_set_dpdf(distr, dpdf_callback)
        print("   ✓ Распределение настроено")
        
        # Пробуем метод TDR (Transformed Density Rejection)
        # Этот метод использует PDF и производную PDF
        print("\n3. Создание генератора методом TDR...")
        par = lib.unur_tdr_new(distr)
        if par == ffi.NULL:
            print("   ⚠ Не удалось создать параметры TDR, пробуем AROU...")
            par = lib.unur_arou_new(distr)
        
        if par == ffi.NULL:
            print("   ✗ Не удалось создать параметры")
            lib.unur_distr_free(distr)
            return False
        
        gen = lib.unur_init(par)
        if gen == ffi.NULL:
            print("   ✗ Не удалось инициализировать генератор")
            # ВАЖНО: если unur_init не удался, нужно освободить par вручную
            lib.unur_par_free(par)
            lib.unur_distr_free(distr)
            return False
        
        print("   ✓ Генератор создан")
        
        # Генерируем выборки
        print("\n4. Генерация выборок...")
        samples = []
        try:
            for i in range(10):
                sample = lib.unur_sample_cont(gen)
                samples.append(sample)
                print(f"   Sample {i+1:2d}: {sample:10.6f}")
        except Exception as e:
            print(f"   ✗ Ошибка: {e}")
            import traceback
            traceback.print_exc()
        
        # Освобождаем память
        print("\n5. Освобождение памяти...")
        lib.unur_free(gen)
        # ВАЖНО: после unur_init(par) параметры автоматически освобождаются
        # Не нужно вызывать unur_par_free(par) - это может привести к двойному освобождению
        lib.unur_distr_free(distr)
        print("   ✓ Память освобождена")
        
        # Важно: callback функции должны оставаться в памяти до освобождения всех объектов
        # Мы сохранили ссылки в списке callbacks, чтобы предотвратить сборку мусора
        
        return len(samples) > 0
        
    except Exception as e:
        print(f"\n✗ Ошибка: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Главная функция"""
    print("\n" + "=" * 70)
    print("ПРОДВИНУТЫЕ ПРИМЕРЫ UNURAN С CALLBACK ФУНКЦИЯМИ")
    print("=" * 70)
    
    examples = [
        ("Генерация с PDF callback", example_with_pdf_callback),
        ("Генерация с PDF и dPDF", example_with_pdf_and_dpdf),
    ]
    
    results = []
    
    for name, func in examples:
        try:
            result = func()
            results.append((name, result))
        except KeyboardInterrupt:
            print("\n\nПрервано пользователем")
            break
        except Exception as e:
            print(f"\n✗ Критическая ошибка: {e}")
            results.append((name, False))
    
    # Итоги
    print("\n" + "=" * 70)
    print("ИТОГИ")
    print("=" * 70)
    
    for name, result in results:
        status = "✓ Успешно" if result else "✗ Ошибка"
        print(f"{status}: {name}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

