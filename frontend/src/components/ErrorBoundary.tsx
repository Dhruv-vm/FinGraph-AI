"use client";

import React from "react";
import { ErrorState, Button } from "./ui";

/** Contains a crash to one view so the shell and other pages keep working. */
export class ViewErrorBoundary extends React.Component<
  { name: string; children: React.ReactNode },
  { error: Error | null }
> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error(`[${this.props.name}] view crashed`, error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="p-6 max-w-2xl">
          <ErrorState
            title={`${this.props.name} failed to render`}
            error={{ message: this.state.error.message }}
            action={
              <Button size="sm" onClick={() => this.setState({ error: null })}>
                Try again
              </Button>
            }
          />
        </div>
      );
    }
    return this.props.children;
  }
}
