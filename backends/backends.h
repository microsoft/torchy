#pragma once

// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

class Trace;

struct TorchyBackend {
  virtual void* compile(const Trace &trace) = 0;
  virtual void run(const void *prog, Trace &trace) = 0;
  virtual void destroy(void *prog) = 0;
};

struct Interpreter final : public TorchyBackend {
  void* compile(const Trace &trace) override { return nullptr; }
  void run(const void *prog, Trace &trace) override;
  void destroy(void *prog) override {}
};

struct TorchScript final : public TorchyBackend {
  void* compile(const Trace &trace) override;
  void run(const void *prog, Trace &trace) override;
  void destroy(void *prog) override;
};
