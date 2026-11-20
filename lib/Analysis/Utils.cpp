//===-- Utils.h -------------------------------------------------*- c++ -*-===//
//
// Part of the Dataflow Scheduler project.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// Implements some common utilities for scheduler analyses.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Analysis/Utils.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>

#include "ktir/Dialect/SpyreOp/SpyreOpTypes.h"

using namespace scheduler;

void scheduler::printLocation(llvm::raw_ostream& OS, mlir::Operation* op) {
  if (!op) {
    OS << "<nullptr op>";
    return;
  }

  if (auto file_loc = op->getLoc()->findInstanceOf<mlir::FileLineColLoc>()) {
    OS << "@ line: " << file_loc.getLine();
  } else {
    OS << "@ line: " << op;
  }
}

auto scheduler::tryGetSizeInBits(mlir::Type type) -> std::optional<unsigned> {
  return llvm::TypeSwitch<mlir::Type, std::optional<unsigned>>(type)
      .Case([](mlir::IntegerType type) -> unsigned { return type.getWidth(); })
      .Case([](mlir::FloatType type) -> unsigned { return type.getWidth(); })
      .Case([](mlir::spyreop::DF16Type) -> unsigned { return 16; })
      .Case([](mlir::VectorType type) -> std::optional<unsigned> {
        if (!type.hasStaticShape()) {
          return std::nullopt;
        }

        auto bits = tryGetSizeInBits(type.getElementType());
        return bits ? std::optional<unsigned>(type.getNumElements() * *bits)
                    : std::nullopt;
      })
      .Default(std::nullopt);
}

auto scheduler::tryGetSizeInBytes(mlir::Type type) -> std::optional<unsigned> {
  const auto maybe_bits = tryGetSizeInBits(type);
  return maybe_bits ? std::optional{(*maybe_bits + 7U) / 8U} : std::nullopt;
}
