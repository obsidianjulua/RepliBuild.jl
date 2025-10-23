# Example: Qt Application

Build a Qt5 application with RepliBuild and use it from Julia.

## Project Overview

Create a Qt5 calculator widget that can be embedded in Julia applications.

## Prerequisites

- Qt5 development files installed
- qmake available

**Install Qt5:**

```bash
# Ubuntu/Debian
sudo apt-get install qt5-default qtbase5-dev

# macOS
brew install qt@5

# Or use RepliBuild's JLL integration (recommended)
```

## Step 1: Initialize Project

```julia
using RepliBuild

RepliBuild.init("qt_calculator")
cd("qt_calculator")
```

## Step 2: Create Qt Project File

Create `calculator.pro`:

```qmake
QT += core gui widgets

TARGET = calculator
TEMPLATE = lib
CONFIG += shared c++11

# Output
DESTDIR = build
OBJECTS_DIR = build/obj
MOC_DIR = build/moc

# Sources
SOURCES += src/calculator_widget.cpp
HEADERS += include/calculator_widget.h

# Include path
INCLUDEPATH += include

# Install
target.path = build
INSTALLS += target
```

## Step 3: Create Calculator Widget

### Header

Create `include/calculator_widget.h`:

```cpp
#ifndef CALCULATOR_WIDGET_H
#define CALCULATOR_WIDGET_H

#include <QWidget>
#include <QPushButton>
#include <QLineEdit>
#include <QGridLayout>

// C API for Julia bindings
extern "C" {
    void* calculator_create();
    void calculator_destroy(void* calc);
    void calculator_show(void* calc);
    void calculator_hide(void* calc);
    const char* calculator_get_display(void* calc);
    void calculator_set_display(void* calc, const char* text);
}

namespace QtCalc {

class CalculatorWidget : public QWidget {
    Q_OBJECT

public:
    explicit CalculatorWidget(QWidget *parent = nullptr);
    ~CalculatorWidget();

    QString getDisplay() const;
    void setDisplay(const QString& text);

private slots:
    void digitClicked();
    void operatorClicked();
    void equalClicked();
    void clearClicked();

private:
    void createButtons();

    QLineEdit* display;
    QGridLayout* layout;

    double leftOperand;
    QString currentOperator;
    bool waitingForOperand;
};

} // namespace QtCalc

#endif
```

### Implementation

Create `src/calculator_widget.cpp`:

```cpp
#include "calculator_widget.h"
#include <QApplication>
#include <cmath>

namespace QtCalc {

CalculatorWidget::CalculatorWidget(QWidget *parent)
    : QWidget(parent), leftOperand(0.0), waitingForOperand(true)
{
    display = new QLineEdit("0");
    display->setReadOnly(true);
    display->setAlignment(Qt::AlignRight);
    display->setMaxLength(15);

    layout = new QGridLayout;
    layout->addWidget(display, 0, 0, 1, 4);

    createButtons();
    setLayout(layout);
    setWindowTitle("Qt Calculator");
}

CalculatorWidget::~CalculatorWidget() {}

QString CalculatorWidget::getDisplay() const {
    return display->text();
}

void CalculatorWidget::setDisplay(const QString& text) {
    display->setText(text);
}

void CalculatorWidget::createButtons() {
    // Digit buttons
    for (int i = 0; i < 10; ++i) {
        QPushButton* button = new QPushButton(QString::number(i));
        connect(button, &QPushButton::clicked, this, &CalculatorWidget::digitClicked);

        if (i == 0) {
            layout->addWidget(button, 4, 0, 1, 2);
        } else {
            int row = 3 - (i - 1) / 3;
            int col = (i - 1) % 3;
            layout->addWidget(button, row, col);
        }
    }

    // Operator buttons
    QPushButton* addButton = new QPushButton("+");
    QPushButton* subButton = new QPushButton("-");
    QPushButton* mulButton = new QPushButton("*");
    QPushButton* divButton = new QPushButton("/");

    connect(addButton, &QPushButton::clicked, this, &CalculatorWidget::operatorClicked);
    connect(subButton, &QPushButton::clicked, this, &CalculatorWidget::operatorClicked);
    connect(mulButton, &QPushButton::clicked, this, &CalculatorWidget::operatorClicked);
    connect(divButton, &QPushButton::clicked, this, &CalculatorWidget::operatorClicked);

    layout->addWidget(addButton, 1, 3);
    layout->addWidget(subButton, 2, 3);
    layout->addWidget(mulButton, 3, 3);
    layout->addWidget(divButton, 4, 3);

    // Special buttons
    QPushButton* equalButton = new QPushButton("=");
    QPushButton* clearButton = new QPushButton("C");

    connect(equalButton, &QPushButton::clicked, this, &CalculatorWidget::equalClicked);
    connect(clearButton, &QPushButton::clicked, this, &CalculatorWidget::clearClicked);

    layout->addWidget(clearButton, 4, 2);
    layout->addWidget(equalButton, 1, 2);
}

void CalculatorWidget::digitClicked() {
    QPushButton* button = qobject_cast<QPushButton*>(sender());
    QString digit = button->text();

    if (waitingForOperand) {
        display->setText(digit);
        waitingForOperand = false;
    } else {
        display->setText(display->text() + digit);
    }
}

void CalculatorWidget::operatorClicked() {
    QPushButton* button = qobject_cast<QPushButton*>(sender());
    QString op = button->text();

    double operand = display->text().toDouble();

    if (!currentOperator.isEmpty()) {
        // Perform previous operation
        if (currentOperator == "+") {
            leftOperand += operand;
        } else if (currentOperator == "-") {
            leftOperand -= operand;
        } else if (currentOperator == "*") {
            leftOperand *= operand;
        } else if (currentOperator == "/") {
            if (operand != 0.0) {
                leftOperand /= operand;
            }
        }
        display->setText(QString::number(leftOperand));
    } else {
        leftOperand = operand;
    }

    currentOperator = op;
    waitingForOperand = true;
}

void CalculatorWidget::equalClicked() {
    double operand = display->text().toDouble();

    if (!currentOperator.isEmpty()) {
        if (currentOperator == "+") {
            leftOperand += operand;
        } else if (currentOperator == "-") {
            leftOperand -= operand;
        } else if (currentOperator == "*") {
            leftOperand *= operand;
        } else if (currentOperator == "/") {
            if (operand != 0.0) {
                leftOperand /= operand;
            }
        }

        display->setText(QString::number(leftOperand));
        currentOperator.clear();
        waitingForOperand = true;
    }
}

void CalculatorWidget::clearClicked() {
    display->setText("0");
    leftOperand = 0.0;
    currentOperator.clear();
    waitingForOperand = true;
}

} // namespace QtCalc

// C API implementation
void* calculator_create() {
    return new QtCalc::CalculatorWidget();
}

void calculator_destroy(void* calc) {
    delete static_cast<QtCalc::CalculatorWidget*>(calc);
}

void calculator_show(void* calc) {
    static_cast<QtCalc::CalculatorWidget*>(calc)->show();
}

void calculator_hide(void* calc) {
    static_cast<QtCalc::CalculatorWidget*>(calc)->hide();
}

const char* calculator_get_display(void* calc) {
    static QString result;
    result = static_cast<QtCalc::CalculatorWidget*>(calc)->getDisplay();
    return result.toUtf8().constData();
}

void calculator_set_display(void* calc, const char* text) {
    static_cast<QtCalc::CalculatorWidget*>(calc)->setDisplay(QString(text));
}
```

## Step 4: Configure RepliBuild

Edit `replibuild.toml`:

```toml
[project]
name = "QtCalculator"
version = "1.0.0"

[build]
system = "qmake"
qt_version = "Qt5"
pro_file = "calculator.pro"
build_dir = "build"

# Or use JLL
use_jll = true
[build.jll_packages]
qt5 = "Qt5Base_jll"

[output]
library_name = "libcalculator"
julia_module_name = "QtCalculator"
libraries = ["build/libcalculator.so"]

[bindings]
# Wrap the C API
export_functions = [
    "calculator_create",
    "calculator_destroy",
    "calculator_show",
    "calculator_hide",
    "calculator_get_display",
    "calculator_set_display"
]
```

## Step 5: Build

```julia
using RepliBuild

# Build Qt project
RepliBuild.build()
```

## Step 6: Generate Julia Wrapper

```julia
# Wrap the compiled library
RepliBuild.wrap_binary("build/libcalculator.so")
```

Or create manual wrapper in `julia/QtCalculator.jl`:

```julia
module QtCalculator

const libcalc = joinpath(@__DIR__, "../build/libcalculator.so")

# C API wrappers
function create()
    ccall((:calculator_create, libcalc), Ptr{Cvoid}, ())
end

function destroy(calc::Ptr{Cvoid})
    ccall((:calculator_destroy, libcalc), Cvoid, (Ptr{Cvoid},), calc)
end

function show(calc::Ptr{Cvoid})
    ccall((:calculator_show, libcalc), Cvoid, (Ptr{Cvoid},), calc)
end

function hide(calc::Ptr{Cvoid})
    ccall((:calculator_hide, libcalc), Cvoid, (Ptr{Cvoid},), calc)
end

function get_display(calc::Ptr{Cvoid})
    ptr = ccall((:calculator_get_display, libcalc), Ptr{UInt8}, (Ptr{Cvoid},), calc)
    unsafe_string(ptr)
end

function set_display(calc::Ptr{Cvoid}, text::String)
    ccall((:calculator_set_display, libcalc), Cvoid, (Ptr{Cvoid}, Ptr{UInt8}), calc, text)
end

# High-level API
mutable struct Calculator
    ptr::Ptr{Cvoid}

    function Calculator()
        ptr = create()
        calc = new(ptr)
        finalizer(c -> destroy(c.ptr), calc)
        return calc
    end
end

Base.show(calc::Calculator) = show(calc.ptr)
Base.hide(calc::Calculator) = hide(calc.ptr)

function get_result(calc::Calculator)
    get_display(calc.ptr)
end

function set_value(calc::Calculator, value::String)
    set_display(calc.ptr, value)
end

export Calculator, get_result, set_value

end # module
```

## Step 7: Use from Julia

```julia
# Ensure Qt application exists
ENV["QT_QPA_PLATFORM"] = "offscreen"  # For headless

using .QtCalculator

# Create calculator
calc = QtCalculator.Calculator()

# Show window
show(calc)

# Get display value
println("Display: ", QtCalculator.get_result(calc))

# Set value
QtCalculator.set_value(calc, "42")
println("New display: ", QtCalculator.get_result(calc))

# Calculator will be automatically destroyed when Julia GC runs
```

## Complete Project Structure

```
qt_calculator/
├── calculator.pro
├── replibuild.toml
├── src/
│   └── calculator_widget.cpp
├── include/
│   └── calculator_widget.h
├── julia/
│   └── QtCalculator.jl
├── build/
│   ├── libcalculator.so
│   ├── moc/
│   └── obj/
└── test/
    └── test_calculator.jl
```

## Next Steps

- Add more calculator features
- Create other Qt widgets
- Build a complete Qt application
- See **[Multi-Module Example](multi-module.md)** for combining with other libraries

## Notes

- Qt requires a running QApplication instance
- Use `QT_QPA_PLATFORM=offscreen` for headless environments
- Qt signals/slots can be exposed through C callbacks
- For complex Qt integration, consider using CxxWrap.jl directly
