defmodule LangChain.ChatModels.ChatMistralAITest do
  alias Langchain.ChatModels.ChatMistralAI
  use LangChain.BaseCase

  alias LangChain.Message
  alias LangChain.MessageDelta
  alias LangChain.Function
  alias LangChain.FunctionParam

  defp hello_world(_args, _context) do
    "Hello world!"
  end

  setup do
    {:ok, hello_world} =
      Function.new(%{
        name: "hello_world",
        description: "Give a hello world greeting.",
        function: fn -> IO.puts("Hello world!") end
      })

    %{hello_world: hello_world}
  end

  describe "new/1" do
    test "works with minimal attr" do
      assert {:ok, %ChatMistralAI{} = mistral_ai} =
               ChatMistralAI.new(%{"model" => "mistral-tiny"})

      assert mistral_ai.model == "mistral-tiny"
    end

    test "returns error when invalid" do
      assert {:error, changeset} = ChatMistralAI.new(%{"model" => nil})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:model]
    end

    test "supports overriding the API endpoint" do
      override_url = "http://localhost:1234/"

      model =
        ChatMistralAI.new!(%{
          "model" => "mistral-tiny",
          "endpoint" => override_url
        })

      assert model.endpoint == override_url
    end
  end

  describe "for_api/3" do
    setup do
      {:ok, mistral_ai} =
        ChatMistralAI.new(%{
          model: "mistral-tiny",
          temperature: 1.0,
          top_p: 1.0,
          max_tokens: 100,
          safe_prompt: true,
          random_seed: 42
        })

      %{mistral_ai: mistral_ai}
    end

    test "generates a map for an API call", %{mistral_ai: mistral_ai} do
      data = ChatMistralAI.for_api(mistral_ai, [], [])

      assert data ==
               %{
                 model: "mistral-tiny",
                 temperature: 1.0,
                 top_p: 1.0,
                 messages: [],
                 stream: false,
                 max_tokens: 100,
                 safe_prompt: true,
                 random_seed: 42
               }
    end

    test "generates a map containing user and assistant messages", %{mistral_ai: mistral_ai} do
      user_message = "Hello Assistant!"
      assistant_message = "Hello User!"

      data =
        ChatMistralAI.for_api(
          mistral_ai,
          [Message.new_user!(user_message), Message.new_assistant!(assistant_message)],
          []
        )

      assert get_in(data, [:messages, Access.at(0), "role"]) == :user

      assert get_in(data, [:messages, Access.at(0), "content"]) == user_message

      assert get_in(data, [:messages, Access.at(1), "role"]) == :assistant

      assert get_in(data, [:messages, Access.at(1), "content"]) == assistant_message
    end
  end

  describe "for_api/1" do
    test "turns a function_call into expected JSON format" do
      msg = Message.new_function_call!("hello_world", "{}")

      json = ChatMistralAI.for_api(msg)

      assert json == %{
               "content" => nil,
               "tool_calls" => [
                 %{
                   "function" => %{"arguments" => "{}", "name" => "hello_world"},
                   "type" => "function"
                 }
               ],
               "role" => :assistant
             }
    end

    test "turns a function_call into expected JSON format with arguments" do
      args = %{"expression" => "11 + 10"}
      msg = Message.new_function_call!("hello_world", Jason.encode!(args))

      json = ChatMistralAI.for_api(msg)

      assert json == %{
               "content" => nil,
               "tool_calls" => [
                 %{
                   "function" => %{
                     "arguments" => "{\"expression\":\"11 + 10\"}",
                     "name" => "hello_world"
                   },
                   "type" => "function"
                 }
               ],
               "role" => :assistant
             }
    end

    test "turns a function response into expected JSON format" do
      msg = Message.new_function!("hello_world", "Hello World!")

      json = ChatMistralAI.for_api(msg)

      assert json == %{"content" => "Hello World!", "name" => "hello_world", "role" => :tool}
    end

    test "works with minimal definition and no parameters" do
      {:ok, fun} = Function.new(%{"name" => "hello_world"})

      result = ChatMistralAI.for_api(fun)
      # result = Function.for_api(fun)

      assert result == %{
               "name" => "hello_world",
               #  NOTE: Sends the required empty parameter definition when none set
               "parameters" => %{"properties" => %{}, "type" => "object"}
             }
    end

    test "supports parameters" do
      params_def = %{
        "type" => "object",
        "properties" => %{
          "p1" => %{"type" => "string"},
          "p2" => %{"description" => "Param 2", "type" => "number"},
          "p3" => %{
            "enum" => ["yellow", "red", "green"],
            "type" => "string"
          }
        },
        "required" => ["p1"]
      }

      {:ok, fun} =
        Function.new(%{
          name: "say_hi",
          description: "Provide a friendly greeting.",
          parameters: [
            FunctionParam.new!(%{name: "p1", type: :string, required: true}),
            FunctionParam.new!(%{name: "p2", type: :number, description: "Param 2"}),
            FunctionParam.new!(%{name: "p3", type: :string, enum: ["yellow", "red", "green"]})
          ]
        })

      # result = Function.for_api(fun)
      result = ChatMistralAI.for_api(fun)

      assert result == %{
               "name" => "say_hi",
               "description" => "Provide a friendly greeting.",
               "parameters" => params_def
             }
    end

    test "supports parameters_schema" do
      params_def = %{
        "type" => "object",
        "properties" => %{
          "p1" => %{"description" => nil, "type" => "string"},
          "p2" => %{"description" => "Param 2", "type" => "number"},
          "p3" => %{
            "description" => nil,
            "enum" => ["yellow", "red", "green"],
            "type" => "string"
          }
        },
        "required" => ["p1"]
      }

      {:ok, fun} =
        Function.new(%{
          "name" => "say_hi",
          "description" => "Provide a friendly greeting.",
          "parameters_schema" => params_def
        })

      # result = Function.for_api(fun)
      result = ChatMistralAI.for_api(fun)

      assert result == %{
               "name" => "say_hi",
               "description" => "Provide a friendly greeting.",
               "parameters" => params_def
             }
    end

    test "does not allow both parameters and parameters_schema" do
      {:error, changeset} =
        Function.new(%{
          name: "problem",
          parameters: [
            FunctionParam.new!(%{name: "p1", type: :string, required: true})
          ],
          parameters_schema: %{stuff: true}
        })

      assert {"Cannot use both parameters and parameters_schema", _} =
               changeset.errors[:parameters]
    end

    test "does not include the function to execute" do
      # don't try and send an Elixir function ref through to the API
      {:ok, fun} = Function.new(%{"name" => "hello_world", "function" => &hello_world/2})
      # result = Function.for_api(fun)
      result = ChatMistralAI.for_api(fun)
      refute Map.has_key?(result, "function")
    end
  end

  describe "do_process_response/2" do
    test "handles receiving a message" do
      response = %{
        "choices" => [
          %{
            "message" => %{
              "role" => "assistant",
              "content" => "Hello User!"
            },
            "finish_reason" => "stop",
            "index" => 0
          }
        ]
      }

      assert [%Message{} = struct] = ChatMistralAI.do_process_response(response)
      assert struct.role == :assistant
      assert struct.content == "Hello User!"
      assert struct.index == 0
      assert struct.status == :complete
    end

    test "errors with invalid role" do
      response = %{
        "choices" => [
          %{
            "message" => %{
              "role" => "unknown role",
              "content" => "Hello User!"
            },
            "finish_reason" => "stop",
            "index" => 0
          }
        ]
      }

      assert [{:error, "role: is invalid"}] = ChatMistralAI.do_process_response(response)
    end

    test "handles receiving MessageDeltas as well" do
      response = %{
        "choices" => [
          %{
            "delta" => %{
              "role" => "assistant",
              "content" => "This is the first part of a mes"
            },
            "finish_reason" => nil,
            "index" => 0
          }
        ]
      }

      assert [%MessageDelta{} = struct] = ChatMistralAI.do_process_response(response)

      assert struct.role == :assistant
      assert struct.content == "This is the first part of a mes"
      assert struct.index == 0
      assert struct.status == :incomplete
    end

    test "handles receiving a tool_calls message" do
      response = %{
        "finish_reason" => "tool_calls",
        "index" => 0,
        "message" => %{
          "tool_calls" => [
            %{
              "function" => %{
                "arguments" => "{}",
                "name" => "hello_world"
              }
            }
          ],
          "role" => "assistant"
        }
      }

      assert %Message{} = struct = ChatMistralAI.do_process_response(response)

      assert struct.role == :assistant
      assert struct.content == nil
      assert struct.function_name == "hello_world"
      assert struct.arguments == %{}
      assert struct.index == 0
    end

    test "handles API error messages" do
      response = %{
        "error" => %{
          "code" => 400,
          "message" => "Invalid request",
          "status" => "INVALID_ARGUMENT"
        }
      }

      assert {:error, error_string} = ChatMistralAI.do_process_response(response)
      assert error_string == "Invalid request"
    end

    test "handles Jason.DecodeError" do
      response = {:error, %Jason.DecodeError{}}

      assert {:error, error_string} = ChatMistralAI.do_process_response(response)
      assert "Received invalid JSON:" <> _ = error_string
    end

    test "handles unexpected response with error" do
      response = %{}
      assert {:error, "Unexpected response"} = ChatMistralAI.do_process_response(response)
    end
  end
end
